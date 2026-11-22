import re
from pathlib import Path
import joblib

# Functions to read a review: this file only DEFINES them and loads the saved models.
# Importing it does not read the dataset, print or save anything.

# ---------- Texts that mean "no criticism" / "nothing positive" (TO REVIEW) ----------
# Rules: lowercase only; the WHOLE text must be equal to one of these (after strip + lowercase).
# "" = empty text.
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

# ---------- Load the saved models ----------

# Main folder of the repository (one level above the folder of this file):
# the paths work from whatever folder the script is launched
ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT / "data"
MODELS_FOLDER = ROOT / "models"

# Load Model 1 (vectorizer, 21 models, thresholds) saved by topic_model.py
saved = joblib.load(MODELS_FOLDER / "topic_model.joblib")
TOPICS = saved["topics"]
vectorizer = saved["vectorizer"]
models = saved["models"]
thresholds = saved["thresholds"]

# Load the sentiment model (vectorizer + logistic regression) saved by sentiment_model.py
# (used only by the "nothing" rule)
saved_sentiment = joblib.load(MODELS_FOLDER / "sentiment_model.joblib")
sentiment_vectorizer = saved_sentiment["vectorizer"]
sentiment_model = saved_sentiment["model"]

# Probability used by the "nothing" rule
NOTHING_RULE_THRESHOLD = 0.90   # P(positive) of the WHOLE text needed to make it empty

# ---------- Functions: cleaning, windows, topics ----------

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

def normalize_text(text):
    """Bring a text to the format of the 515K dataset: only letters and numbers,
    separated by single spaces ("Can't fault it." -> "Can t fault it")."""
    if not isinstance(text, str):          # a missing value (NaN) counts as empty
        return ""
    # [\W_] = any character that is not a letter or a number (accented letters are kept)
    text = re.sub(r"[\W_]", " ", text)
    return " ".join(text.split())          # no double spaces, no spaces at the borders

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

def clean_negative(text, use_nothing_rule=False):
    """Format of the 515K + rule 1 (exact placeholders) + rule 2 (short expressions) for a
    negative text. If use_nothing_rule is True, also the "nothing" rule (normally it stays off)."""
    text = normalize_text(text)
    text = remove_placeholder(text, EMPTY_NEGATIVE)
    text = remove_short_expression(text, EMPTY_NEGATIVE_EXPRESSIONS)
    if use_nothing_rule:
        text = remove_nothing_praise(text)
    return text

def clean_positive(text):
    """Format of the 515K + rule 1 (exact placeholders) for a positive text."""
    text = normalize_text(text)
    return remove_placeholder(text, EMPTY_POSITIVE)
