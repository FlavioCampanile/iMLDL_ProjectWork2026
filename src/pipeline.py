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


# ---------- Reviews written as ONE text (Google, TripAdvisor...): the fixed rule ----------
# Booking divides a review into "liked" and "disliked"; most other platforms do not.
# The fixed rule splits a whole review into pieces, Model 2 sends each piece to the
# negative or to the positive side, and the uncertain pieces are put aside.
# After that, the two sides go into the report exactly like the two Booking fields.

# Words that usually turn the judgement around: the text is cut BEFORE them
# ("The room was small but very clean" -> "The room was small" / "but very clean")
CONTRAST_WORDS = ["but", "however", "although", "though", "except", "whereas", "unfortunately"]
MIN_PIECE_WORDS = 2       # a shorter piece ("though") is glued back to the piece before it
MIN_COMMA_PART_WORDS = 2  # same idea for the parts made by a comma

# Model 2 probability of "positive": above HIGH -> positive side, below LOW -> negative side,
# in between -> put aside (the model is not sure)
LOW_POSITIVE = 0.35
HIGH_POSITIVE = 0.65

def glue_short_pieces(pieces, min_words):
    """Glue every piece with fewer than `min_words` words to the piece before it
    (the first piece, if short, is glued to the next one)."""
    result = []
    for piece in pieces:
        if len(piece.split()) < min_words and len(result) > 0:
            result[-1] = result[-1] + " " + piece
        else:
            result.append(piece)
    # The first piece was short and nothing came before it: glue it to the second one
    if len(result) >= 2 and len(result[0].split()) < min_words:
        result[1] = result[0] + " " + result[1]
        result = result[1:]
    return result

def split_review(text):
    """Split ONE whole review into pieces, each with (ideally) one judgement.
    1) cut on . ! ? ; and on new lines (= sentences)
    2) inside a sentence, cut before a contrast word (but, however, ...)
    3) cut on the commas only if the piece talks about two or more topics (Model 1)
    The punctuation is still there: this must run BEFORE normalize_text."""
    if not isinstance(text, str):              # a missing value (NaN) counts as empty
        return []
    pieces = []
    # Step 1: sentences. [.!?;\n]+ = one or more of these characters in a row
    for sentence in re.split(r"[.!?;\n]+", text):
        if sentence.strip() == "":
            continue
        # Step 2: cut before each contrast word (\b = word boundary, so "but" never matches "button";
        # (?=...) cuts BEFORE the word and keeps it in the next piece)
        pattern = r"\b(?=(?:" + "|".join(CONTRAST_WORDS) + r")\b)"
        parts = [p.strip(" ,") for p in re.split(pattern, sentence, flags=re.IGNORECASE)]
        parts = [p for p in parts if p != ""]
        pieces.extend(glue_short_pieces(parts, MIN_PIECE_WORDS))

    # Step 3: commas, only in pieces with two or more topics ("Great location, rude staff")
    # (Model 1 runs only on the pieces that have a comma: the others are never cut here)
    result = []
    for piece in pieces:
        if "," in piece and len(predict_topics([normalize_text(piece)])[0]) >= 2:
            parts = [p.strip() for p in piece.split(",") if p.strip() != ""]
            result.extend(glue_short_pieces(parts, MIN_COMMA_PART_WORDS))
        else:
            result.append(piece)
    return result

def route_pieces(pieces, low=LOW_POSITIVE, high=HIGH_POSITIVE):
    """Send each piece to a side with Model 2. Return three lists of pieces:
    negative, positive, put aside (probability of "positive" between low and high)."""
    negative, positive, aside = [], [], []
    if len(pieces) == 0:
        return negative, positive, aside
    p_positive = sentiment_model.predict_proba(sentiment_vectorizer.transform(pieces))[:, 1]
    for piece, p in zip(pieces, p_positive):
        if p >= high:
            positive.append(piece)
        elif p <= low:
            negative.append(piece)
        else:
            aside.append(piece)
    return negative, positive, aside

def split_full_text(text):
    """ONE whole review -> (negative text, positive text, number of pieces, number put aside).
    Every piece is cleaned on its own first, so "No complaints" inside a longer review
    is still recognized as "no criticism" (same rules as the Booking fields)."""
    pieces = split_review(text)
    negative, positive, aside = route_pieces(pieces)
    negative = [clean_negative(p) for p in negative]
    positive = [clean_positive(p) for p in positive]
    negative_text = " ".join(p for p in negative if p != "")
    positive_text = " ".join(p for p in positive if p != "")
    return negative_text, positive_text, len(pieces), len(aside)
