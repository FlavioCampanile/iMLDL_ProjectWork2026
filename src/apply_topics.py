import re
import time
import joblib
import pandas as pd

# ---------- Texts that mean "no criticism" / "nothing positive" (TO REVIEW) ----------
# Rules: lowercase only; the WHOLE text must be equal to one of these (after strip + lowercase).
# "" = empty text. NOT used yet: we will use them in the next step.
EMPTY_NEGATIVE = [
    "", "no negative", "no negatives", "nothing", "no thing", "none", "nil", "n a", "na",
    "not applicable", "no", "nothing really", "nothing at all", "absolutely nothing",
    "nothing in particular", "nothing to report", "nothing to say", "nothing to mention",
    "nothing comes to mind", "nothing i can think of", "can t think of anything",
    "no complaints", "no complaints at all", "nothing to complain about", "nothing to complain",
    "no issues", "no problems", "no dislikes", "no comment", "no comments", "no bad experience",
    "nothing bad", "nothing bad to say", "nothing to dislike", "there was nothing to dislike",
    "nothing to not like", "nothing not to like", "there was nothing not to like",
    "nothing i didn t like", "there was nothing i didn t like", "nothing we didn t like",
    "there was nothing we didn t like", "nothing could have been better",
    "all good", "it was all good", "nothing all good", "nothing it was all good",
    "everything was perfect", "nothing it was perfect", "nothing everything was perfect",
    "nothing everything was great", "everything was great", "leaving",
]
# Expressions that make a NEGATIVE text empty, but ONLY if the text has at most 8 words (TO REVIEW).
# Lowercase only; the expression must appear as whole words anywhere in the text.
# NOT used yet.
EMPTY_NEGATIVE_EXPRESSIONS = [
    # "can't fault"
    "can t fault", "couldn t fault", "could not fault", "cannot fault",
    # "no complaints"
    "nothing to complain", "no complaints", "no complaint",
    # "nothing I didn't like"
    "nothing to dislike", "nothing not to like", "nothing i didn t like", "nothing we didn t like",
    "nothing i did not like", "nothing we did not like",
    # "liked / loved everything"
    "liked everything", "loved everything", "liked it all", "loved it all", "liked all",
    # "everything was good"
    "everything was good", "everything was fine", "everything was great",
    "everything was excellent", "everything was perfect", "everything was wonderful",
    "everything was very good", "all was great", "all perfect", "it was all great",
    # "nothing (bad)"
    "nothing bad", "nothing to report", "really nothing",
]
MAX_WORDS_FOR_EXPRESSIONS = 8
# If the text contains one of these words, it is NEVER made empty ("nothing bad except the noise")
EXCEPTION_WORDS = ["but", "except", "apart", "only", "however", "although", "besides"]

EMPTY_POSITIVE = [
    "", "no positive", "nothing", "none", "n a", "na", "no",
    "nothing really", "nothing at all",
]

# ---------- Step 1: load the 515K dataset and look at the most frequent texts ----------

# Load ALL the columns
df = pd.read_csv("Data/Hotel_Reviews.csv")
print("Rows in the file:", len(df))

# Remove duplicate rows (identical in ALL the columns), keeping the first
df = df.drop_duplicates()

# Only now keep the 4 columns we need
COLUMNS = ["Hotel_Address", "Negative_Review", "Positive_Review", "Reviewer_Score"]
df = df[COLUMNS].copy()

# Remove spaces at the start and at the end of the texts (" No Negative " -> "No Negative")
for col in ["Hotel_Address", "Negative_Review", "Positive_Review"]:
    df[col] = df[col].str.strip()

print("Number of rows:", len(df))

# The 30 most frequent texts in each column, with how many times they appear
for col in ["Negative_Review", "Positive_Review"]:
    print()
    print(f"30 most frequent texts in {col}:")
    counts = df[col].value_counts().head(30)
    for text, count in counts.items():
        print(f"  {count:>8}  {text}")

# ---------- Step 2: placeholders, windows of words, topics of a whole text ----------

# Load Model 1 (vectorizer, 21 models, thresholds) saved by topic_model.py
saved = joblib.load("models/topic_model.joblib")
TOPICS = saved["topics"]
vectorizer = saved["vectorizer"]
models = saved["models"]
thresholds = saved["thresholds"]

# Load the sentiment model (vectorizer + logistic regression) saved by sentiment_model.py
# (used only by the "nothing" rule)
saved_sentiment = joblib.load("models/sentiment_model.joblib")
sentiment_vectorizer = saved_sentiment["vectorizer"]
sentiment_model = saved_sentiment["model"]

# "Nothing" rule: SWITCHED OFF (table in Data/review_topics.csv).
# Put True to build the second table (Data/review_topics_nothing.csv), used by impact_nothing.py
USE_NOTHING_RULE = False
NOTHING_RULE_THRESHOLD = 0.90   # P(positive) of the WHOLE text needed to make it empty

def predict_topics(texts):
    """Given a list of sentences, return for each one the list of topics above threshold
    (same function as in topic_model.py)."""
    texts_tfidf = vectorizer.transform(texts)
    results = [[] for _ in texts]
    for topic in TOPICS:
        probs = models[topic].predict_proba(texts_tfidf)[:, 1]
        for i, p in enumerate(probs):
            if p >= thresholds[topic]:
                results[i].append(topic)
    return results

def remove_placeholder(text, empty_list):
    """Return "" if the text (in lowercase) is one of the placeholders, otherwise the text."""
    if not isinstance(text, str):          # a missing value (NaN) counts as empty
        return ""
    if text.lower() in empty_list:
        return ""
    return text

def remove_short_expression(text, expressions, max_words=MAX_WORDS_FOR_EXPRESSIONS):
    """Return "" if the text has at most `max_words` words, contains NO exception word
    AND contains one of the expressions (as whole words); otherwise return the text unchanged."""
    if len(text.split()) > max_words:      # long text: it probably also contains a real criticism
        return text
    lower = text.lower()
    for word in lower.split():             # one of the words is "but", "except", ...: keep the text
        if word in EXCEPTION_WORDS:
            return text
    for expr in expressions:
        # \b = word boundary: "liked all" matches "i liked all" but not "liked allocation"
        if re.search(r"\b" + re.escape(expr) + r"\b", lower):
            return ""
    return text

def remove_nothing_praise(text, threshold=NOTHING_RULE_THRESHOLD):
    """NEGATIVE field only: return "" if the text starts with "nothing" AND the whole text
    is clearly positive for the sentiment model ("Nothing everything was perfect the staff...");
    otherwise return the text unchanged."""
    words = text.lower().split()
    if len(words) == 0 or words[0] != "nothing":
        return text
    p_positive = sentiment_model.predict_proba(sentiment_vectorizer.transform([text]))[0, 1]
    if p_positive >= threshold:
        return ""
    return text

def make_windows(text, size=12, step=6):
    """Split a text into windows of `size` words that overlap (words 1-12, 7-18, 13-24...)."""
    words = text.split()                   # list of the words, split on spaces
    if len(words) == 0:                    # empty text: no windows
        return []
    windows = []
    start = 0
    while True:
        windows.append(" ".join(words[start:start + size]))   # glue the words back with spaces
        if start + size >= len(words):     # this window reaches the end of the text: stop
            break
        start += step
    return windows

def topics_of_text(text):
    """Apply predict_topics to all the windows of a text; return the topics found in at least one."""
    windows = make_windows(text)
    if len(windows) == 0:
        return []
    found = set()                          # a set keeps each topic only once
    for window_topics in predict_topics(windows):
        found.update(window_topics)
    return [topic for topic in TOPICS if topic in found]   # same order as TOPICS

# Try on 5 random reviews (random_state fixes which ones)
print()
print("=== Topics found in 5 random reviews ===")
for _, row in df.sample(5, random_state=42).iterrows():
    negative = remove_placeholder(row["Negative_Review"], EMPTY_NEGATIVE)
    positive = remove_placeholder(row["Positive_Review"], EMPTY_POSITIVE)
    print()
    print(f"Score: {row['Reviewer_Score']}")
    print(f"  NEGATIVE: {negative!r}")
    print(f"    -> {topics_of_text(negative)}")
    print(f"  POSITIVE: {positive!r}")
    print(f"    -> {topics_of_text(positive)}")

# ---------- Step 2b: which negative texts get ONLY "Generic"? ----------

def topics_of_many_texts(texts):
    """Like topics_of_text, but for a whole list of texts at once (much faster):
    all the windows of all the texts go to predict_topics in ONE call."""
    all_windows = []                       # the windows of all the texts, one after the other
    owner = []                             # owner[k] = index of the text that window k comes from
    for i, text in enumerate(texts):
        for window in make_windows(text):
            all_windows.append(window)
            owner.append(i)
    found = [set() for _ in texts]         # one empty set of topics per text
    if len(all_windows) > 0:
        for k, window_topics in enumerate(predict_topics(all_windows)):
            found[owner[k]].update(window_topics)
    return [[topic for topic in TOPICS if topic in f] for f in found]

sample = df.sample(20000, random_state=42)
neg_texts = [remove_placeholder(t, EMPTY_NEGATIVE) for t in sample["Negative_Review"]]
neg_topics = topics_of_many_texts(neg_texts)

# Keep the texts whose ONLY topic is Generic, in lowercase so "Nothing" and "nothing" are counted together
only_generic = [text.lower() for text, topics in zip(neg_texts, neg_topics) if topics == ["Generic"]]
print()
print("Negative texts with ONLY Generic (out of 20000 reviews):", len(only_generic))
print("40 most frequent:")
for text, count in pd.Series(only_generic).value_counts().head(40).items():
    print(f"  {count:>5}  {text}")

# ---------- Step 2c: test and apply the second rule (short expressions) ----------

# Test on 5 sentences: only the first one should become empty
tests = [
    "can t fault anything about this hotel",
    "can t fault the staff but the room was noisy",
    "nothing bad except the noise",
    "liked all of it except the bed",
    "Everything was wrong with this hotel",
]
print()
print("Test of the second rule:")
for t in tests:
    result = remove_short_expression(t, EMPTY_NEGATIVE_EXPRESSIONS)
    print(f"  {t!r:<50} -> {result!r}")

# Apply the rule to the same 20000 negative texts
neg_texts_rule = [remove_short_expression(t, EMPTY_NEGATIVE_EXPRESSIONS) for t in neg_texts]
n_emptied = sum(1 for before, after in zip(neg_texts, neg_texts_rule) if before != "" and after == "")
print()
print("Negative texts made empty by the second rule:", n_emptied)

# Topics after the rule: an emptied text has no topics, the other texts did not change
neg_topics_rule = [[] if after == "" else topics for after, topics in zip(neg_texts_rule, neg_topics)]
only_generic_rule = [text.lower() for text, topics in zip(neg_texts_rule, neg_topics_rule)
                     if topics == ["Generic"]]
print("Negative texts with ONLY Generic: before =", len(only_generic), "- after =", len(only_generic_rule))
print("40 most frequent after the rule:")
for text, count in pd.Series(only_generic_rule).value_counts().head(40).items():
    print(f"  {count:>5}  {text}")

# ---------- Step 3: build the Model 3 table (test on 5000 random reviews) ----------

def clean_negative(text):
    """Rule 1 (exact placeholders) + rule 2 (short expressions) for a negative text.
    If USE_NOTHING_RULE is True, also the "nothing" rule."""
    text = remove_placeholder(text, EMPTY_NEGATIVE)
    text = remove_short_expression(text, EMPTY_NEGATIVE_EXPRESSIONS)
    if USE_NOTHING_RULE:
        text = remove_nothing_praise(text)
    return text

def clean_positive(text):
    """Rule 1 (exact placeholders) for a positive text."""
    return remove_placeholder(text, EMPTY_POSITIVE)

def build_table(data):
    """Return the Model 3 table: Hotel_Address, Reviewer_Score, 21 Neg_ and 21 Pos_ columns (0/1),
    plus the cleaned texts and the topic lists (useful to check the results by eye)."""
    neg_texts = [clean_negative(t) for t in data["Negative_Review"]]
    pos_texts = [clean_positive(t) for t in data["Positive_Review"]]
    neg_topics = topics_of_many_texts(neg_texts)
    pos_topics = topics_of_many_texts(pos_texts)

    table = pd.DataFrame({
        "Hotel_Address": data["Hotel_Address"].values,
        "Reviewer_Score": data["Reviewer_Score"].values,
    })
    for topic in TOPICS:
        # 1 if the topic is in the list of that review, otherwise 0
        table["Neg_" + topic] = [1 if topic in topics else 0 for topics in neg_topics]
    for topic in TOPICS:
        table["Pos_" + topic] = [1 if topic in topics else 0 for topics in pos_topics]
    return table, neg_texts, pos_texts, neg_topics, pos_topics

sample = df.sample(5000, random_state=42)
start_time = time.perf_counter()
table, neg_texts, pos_texts, neg_topics, pos_topics = build_table(sample)
seconds = time.perf_counter() - start_time

print()
print(f"Table built for {len(table)} reviews in {seconds:.1f} seconds")
print(f"Estimated time for all {len(df)} reviews: {seconds * len(df) / len(table) / 60:.1f} minutes")

# Quick check of the "nothing" rule on two sentences
example = "Nothing everything was perfect breakfast staff were so polite and the food was lovely"
print()
print(f"'nothing' rule test: {example!r} -> {remove_nothing_praise(example)!r}")
example = "Nothing works in the room the shower is broken"
print(f"'nothing' rule test: {example!r} -> {remove_nothing_praise(example)!r}")

# ---------- Step 4: apply everything to ALL the reviews and save the table ----------

start_time = time.perf_counter()
full_table, _, _, _, _ = build_table(df)
minutes = (time.perf_counter() - start_time) / 60
print()
print(f"Table built for ALL {len(full_table)} reviews in {minutes:.1f} minutes")

# Two different files, so the table WITH the rule never overwrites the original one
if USE_NOTHING_RULE:
    output_file = "Data/review_topics_nothing.csv"
else:
    output_file = "Data/review_topics.csv"

# Save the table (index=False: do not save the row numbers as an extra column)
full_table.to_csv(output_file, index=False)
print("Saved to", output_file, "-", full_table.shape[0], "rows,", full_table.shape[1], "columns")

# Percentage of reviews that criticize / praise each topic, on the whole dataset
print()
print(f"{'Topic':<28}{'% Neg':>8}{'% Pos':>8}")
for topic in TOPICS:
    pct_neg = full_table["Neg_" + topic].mean() * 100     # mean of a 0/1 column = share of 1s
    pct_pos = full_table["Pos_" + topic].mean() * 100
    print(f"{topic:<28}{pct_neg:>8.1f}{pct_pos:>8.1f}")

# How many reviews have no criticism / no praise at all
neg_cols = ["Neg_" + topic for topic in TOPICS]
pos_cols = ["Pos_" + topic for topic in TOPICS]
print()
print("Reviews with 0 Neg_ topics:", (full_table[neg_cols].sum(axis=1) == 0).sum())
print("Reviews with 0 Pos_ topics:", (full_table[pos_cols].sum(axis=1) == 0).sum())
