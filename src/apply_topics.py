import time
import pandas as pd
from pipeline import (DATA_FOLDER, TOPICS, EMPTY_NEGATIVE, EMPTY_POSITIVE, EMPTY_NEGATIVE_EXPRESSIONS,
                      remove_placeholder, remove_short_expression, remove_nothing_praise,
                      topics_of_text, topics_of_many_texts, clean_negative, clean_positive)

# "Nothing" rule: SWITCHED OFF (table in data/review_topics.csv).
# Put True to build the second table (data/review_topics_nothing.csv), used by impact_nothing.py
USE_NOTHING_RULE = False

# ---------- Step 1: load the 515K dataset and look at the most frequent texts ----------

# Load ALL the columns
df = pd.read_csv(DATA_FOLDER / "Hotel_Reviews.csv")
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

# ---------- Step 2: topics of a whole text (the functions are in pipeline.py) ----------

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

def build_table(data):
    """Return the Model 3 table: Hotel_Address, Reviewer_Score, 21 Neg_ and 21 Pos_ columns (0/1),
    plus the cleaned texts and the topic lists (useful to check the results by eye)."""
    neg_texts = [clean_negative(t, USE_NOTHING_RULE) for t in data["Negative_Review"]]
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
    output_file = DATA_FOLDER / "review_topics_nothing.csv"
else:
    output_file = DATA_FOLDER / "review_topics.csv"

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
