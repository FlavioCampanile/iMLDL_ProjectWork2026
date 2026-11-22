import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Main folder of the repository (one level above the folder of this file):
# the paths work from whatever folder the script is launched
ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT / "data"
MODELS_FOLDER = ROOT / "models"

# ---------- Keyword lists for the keyword baseline (written by hand) ----------
# Rules: lowercase only; a keyword must match a WHOLE word in the sentence
# (so "bed" does NOT match "beds": plurals are listed explicitly);
# a keyword can be more than one word (e.g. "check in").
KEYWORDS = {
    "Clean": ["clean", "cleaned", "cleaning", "cleanliness", "dirty", "dust", "dusty",
              "stain", "stains", "stained", "smell", "smelled", "smelly", "spotless",
              "hygiene", "filthy", "mould", "mold"],
    "Comfort": ["comfortable", "uncomfortable", "comfy", "comfort", "cozy", "cosy", "cramped"],
    "Facilities/Amenities": ["facilities", "facility", "amenities", "gym", "spa", "sauna",
                             "fitness", "kettle", "fridge", "minibar", "hairdryer", "tv",
                             "television", "air conditioning", "aircon"],
    "Location": ["location", "located", "central", "centre", "center", "walking distance",
                 "metro", "station", "nearby", "area", "neighbourhood", "neighborhood",
                 "transport", "airport"],
    "Restaurant (dinner)": ["restaurant", "restaurants", "dinner", "dinners", "meal", "meals",
                            "menu", "chef", "food", "lunch", "cuisine", "dishes"],
    "Staff": ["staff", "reception", "helpful", "friendly", "rude", "polite", "service",
              "personnel", "employees", "manager", "waiter", "waiters", "waitress",
              "housekeeping", "team"],
    "View (Balcony)": ["view", "views", "balcony", "balconies", "terrace", "overlooking",
                       "panoramic"],
    "Breakfast": ["breakfast", "breakfasts", "buffet", "coffee", "eggs", "pastries",
                  "croissant", "croissants", "bread", "fruit", "juice"],
    "Room": ["room", "rooms", "suite", "suites", "spacious", "decor", "furniture",
             "wardrobe", "closet"],
    "Pool": ["pool", "pools", "swimming", "swim", "sunbeds", "loungers", "jacuzzi"],
    "Beach": ["beach", "beaches", "sea", "sand", "sandy", "seaside", "shore"],
    "Bathroom/Shower (toilet)": ["bathroom", "bathrooms", "shower", "showers", "toilet",
                                 "toilets", "bath", "bathtub", "tub", "sink", "towel",
                                 "towels", "hot water"],
    "Bar": ["bar", "bars", "drink", "drinks", "cocktail", "cocktails", "beer", "wine",
            "bartender", "pub"],
    "Bed": ["bed", "beds", "mattress", "mattresses", "pillow", "pillows", "sheets",
            "linen", "bedding"],
    "Parking": ["parking", "park", "parked", "car", "cars", "garage", "valet"],
    "Noise": ["noise", "noises", "noisy", "loud", "quiet", "hear", "heard", "soundproof",
              "soundproofing", "traffic", "thin walls"],
    "Reception-checkin": ["reception", "receptionist", "receptionists", "check in",
                          "check-in", "checkin", "checked in", "check out", "check-out",
                          "checkout", "front desk", "arrival", "key", "keys"],
    "Lift": ["lift", "lifts", "elevator", "elevators"],
    "Value for money": ["price", "prices", "priced", "value", "money", "expensive", "cheap",
                        "overpriced", "worth", "cost", "costs", "affordable", "euro", "euros"],
    "Wi-Fi": ["wifi", "wi-fi", "internet", "connection", "signal"],
    "Generic": ["recommend", "recommended", "overall",
                "experience", "place", "again"],
}

# The 21 topic columns (one sentence can have more than one topic = multi-label)
TOPICS = [
    "Clean", "Comfort", "Facilities/Amenities", "Location", "Restaurant (dinner)",
    "Staff", "View (Balcony)", "Breakfast", "Room", "Pool", "Beach",
    "Bathroom/Shower (toilet)", "Bar", "Bed", "Parking", "Noise",
    "Reception-checkin", "Lift", "Value for money", "Wi-Fi", "Generic",
]

# ---------- Step 1: load and clean the dataset ----------

# Load the dataset (utf-8-sig removes the BOM at the start of the file)
df = pd.read_csv(DATA_FOLDER / "HRAST.csv", encoding="utf-8-sig")

# Remove the empty column without a name
df = df.drop(columns=["Unnamed: 2"])

# Convert the topic columns to integers: a strange value (e.g. "σ") becomes 0
for topic in TOPICS:
    df[topic] = pd.to_numeric(df[topic], errors="coerce").fillna(0).astype(int)

# Remove duplicate sentences (same text in "review"), keeping the first
df = df.drop_duplicates(subset="review")

# Number of topics of each sentence (sum of the 21 columns along the row)
n_topics = df[TOPICS].sum(axis=1)

print("Number of sentences:", len(df))
print("Sentences with 0 topics:", (n_topics == 0).sum())
print("Sentences with 2 or more topics:", (n_topics >= 2).sum())

# How many times each topic appears, from the most to the least frequent
print()
print("Topic frequency:")
counts = df[TOPICS].sum().sort_values(ascending=False)
for topic, count in counts.items():
    print(f"  {topic:<28}{count:>6}")

# ---------- Step 2: train / validation / test split ----------

# X = the texts (input), Y = the 21 topic columns (correct answers, one 0/1 per topic)
X = df["review"]
Y = df[TOPICS]

# First split: 15% of all sentences become the test set
X_rest, X_test, Y_rest, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)

# Second split: from the remaining 85%, take 15% of the TOTAL as validation (0.15 / 0.85 of the rest)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_rest, Y_rest, test_size=0.15 / 0.85, random_state=42
)

print()
print("Train sentences:", len(X_train))
print("Validation sentences:", len(X_val))
print("Test sentences:", len(X_test))

# ---------- Step 3: two baselines, evaluated on the TEST set ----------

def evaluate(name, Y_true, Y_pred):
    """Print F1 for each topic, micro F1 and macro F1."""
    # average=None gives one F1 per topic; zero_division=0 means "F1 = 0 if a topic is never predicted"
    f1_per_topic = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    print()
    print(f"=== {name} ===")
    for topic, f1 in zip(TOPICS, f1_per_topic):
        print(f"  {topic:<28}{f1:>8.4f}")
    print(f"  {'MICRO F1':<28}{micro:>8.4f}")
    print(f"  {'MACRO F1':<28}{macro:>8.4f}")
    return micro, macro

# Baseline 1 - Dummy: every sentence gets ONLY the most frequent topic (Room)
Y_pred_dummy = np.zeros(Y_test.shape, dtype=int)   # a table of 0s, same size as Y_test
Y_pred_dummy[:, TOPICS.index("Room")] = 1          # put 1 in the whole "Room" column
dummy_scores = evaluate("Dummy (always Room)", Y_test, Y_pred_dummy)

# Baseline 2 - Keywords: a topic is 1 if the sentence contains at least one of its keywords
def has_keyword(sentence, words):
    """True if the sentence contains at least one of the words (as a whole word)."""
    sentence = sentence.lower()
    for word in words:
        # \b = word boundary: "bar" matches "the bar is" but not "barely"
        if re.search(r"\b" + re.escape(word) + r"\b", sentence):
            return True
    return False

Y_pred_kw = np.zeros(Y_test.shape, dtype=int)
for j, topic in enumerate(TOPICS):                 # j = column number, topic = column name
    for i, sentence in enumerate(X_test):          # i = row number, sentence = the text
        if has_keyword(sentence, KEYWORDS[topic]):
            Y_pred_kw[i, j] = 1
keyword_scores = evaluate("Keywords", Y_test, Y_pred_kw)

# Comparison table
print()
print(f"{'Model':<22}{'Micro F1':>10}{'Macro F1':>10}")
print(f"{'Dummy (Room)':<22}{dummy_scores[0]:>10.4f}{dummy_scores[1]:>10.4f}")
print(f"{'Keywords':<22}{keyword_scores[0]:>10.4f}{keyword_scores[1]:>10.4f}")

# ---------- Step 4: TF-IDF + one logistic regression per topic ----------

# TF-IDF: learns the vocabulary ONLY from the train set, then transforms validation and test
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Tables where we save the probabilities: one row per sentence, one column per topic
P_val = np.zeros(Y_val.shape)
P_test = np.zeros(Y_test.shape)

# 21 separate models: each one answers only "does this sentence talk about THIS topic?"
models = {}
for j, topic in enumerate(TOPICS):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, Y_train[topic])          # learns from the column of this topic only
    models[topic] = model
    # predict_proba gives [prob. of 0, prob. of 1] for each sentence: we keep column 1
    P_val[:, j] = model.predict_proba(X_val_tfidf)[:, 1]
    P_test[:, j] = model.predict_proba(X_test_tfidf)[:, 1]

# For now: "yes" if the probability is at least 0.5 (same threshold for every topic)
Y_pred_model = (P_test >= 0.5).astype(int)
model_scores = evaluate("TF-IDF + LogReg (threshold 0.5)", Y_test, Y_pred_model)

# Comparison table with all the models so far
print()
print(f"{'Model':<22}{'Micro F1':>10}{'Macro F1':>10}")
print(f"{'Dummy (Room)':<22}{dummy_scores[0]:>10.4f}{dummy_scores[1]:>10.4f}")
print(f"{'Keywords':<22}{keyword_scores[0]:>10.4f}{keyword_scores[1]:>10.4f}")
print(f"{'TF-IDF + LogReg 0.5':<22}{model_scores[0]:>10.4f}{model_scores[1]:>10.4f}")

# ---------- Step 5: one threshold per topic, chosen on the VALIDATION set ----------

# Candidate thresholds: 0.05, 0.10, ..., 0.90 (round avoids tiny decimals like 0.15000000000000002)
candidates = np.round(np.arange(1, 19) * 0.05, 2)

best_thresholds = np.zeros(len(TOPICS))
for j, topic in enumerate(TOPICS):
    best_f1 = -1
    for t in candidates:
        # F1 of this topic on VALIDATION if we say "yes" when probability >= t
        f1 = f1_score(Y_val[topic], (P_val[:, j] >= t).astype(int), zero_division=0)
        if f1 > best_f1:                   # keep the threshold with the highest F1
            best_f1 = f1
            best_thresholds[j] = t

# Apply the 21 chosen thresholds to the TEST set (each column compared with its own threshold)
Y_pred_tuned = (P_test >= best_thresholds).astype(int)

# F1 per topic on the test set, for keywords and for the tuned model
f1_kw_topic = f1_score(Y_test, Y_pred_kw, average=None, zero_division=0)
f1_tuned_topic = f1_score(Y_test, Y_pred_tuned, average=None, zero_division=0)
n_test_topic = Y_test.sum()                # how many test sentences have each topic

print()
print("=== Per-topic results on the TEST set ===")
print(f"{'Topic':<28}{'N test':>8}{'Thresh.':>9}{'F1 kw':>9}{'F1 model':>10}")
for j, topic in enumerate(TOPICS):
    print(f"{topic:<28}{n_test_topic[topic]:>8}{best_thresholds[j]:>9.2f}"
          f"{f1_kw_topic[j]:>9.4f}{f1_tuned_topic[j]:>10.4f}")

# Micro and macro F1 of all the models
tuned_micro = f1_score(Y_test, Y_pred_tuned, average="micro", zero_division=0)
tuned_macro = f1_score(Y_test, Y_pred_tuned, average="macro", zero_division=0)

print()
print(f"{'Model':<26}{'Micro F1':>10}{'Macro F1':>10}")
print(f"{'Dummy (Room)':<26}{dummy_scores[0]:>10.4f}{dummy_scores[1]:>10.4f}")
print(f"{'Keywords':<26}{keyword_scores[0]:>10.4f}{keyword_scores[1]:>10.4f}")
print(f"{'TF-IDF + LogReg 0.5':<26}{model_scores[0]:>10.4f}{model_scores[1]:>10.4f}")
print(f"{'TF-IDF + LogReg tuned':<26}{tuned_micro:>10.4f}{tuned_macro:>10.4f}")

# ---------- Step 6: top words, prediction function, saving the model ----------

# Top 10 words (highest weight) for 4 topics
words = vectorizer.get_feature_names_out()
for topic in ["Staff", "Clean", "Noise", "Generic"]:
    weights = models[topic].coef_[0]          # one weight per word in the vocabulary
    order = weights.argsort()[::-1]           # word indices from the highest weight to the lowest
    print()
    print(f"10 words with the highest weight for {topic}:")
    for i in order[:10]:
        print(f"  {words[i]:<20}{weights[i]:>8.3f}")

# Thresholds as a dictionary {topic: threshold}, easier to read
thresholds = {topic: float(best_thresholds[j]) for j, topic in enumerate(TOPICS)}

def predict_topics(texts):
    """Given a list of sentences, return for each one the list of topics above threshold."""
    texts_tfidf = vectorizer.transform(texts)       # same vocabulary learned on the train set
    results = [[] for _ in texts]                   # one empty list of topics per sentence
    for topic in TOPICS:
        probs = models[topic].predict_proba(texts_tfidf)[:, 1]
        for i, p in enumerate(probs):
            if p >= thresholds[topic]:
                results[i].append(topic)
    return results

# Try the function on 3 invented sentences
examples = [
    "The room was tiny but the staff were lovely",
    "Breakfast was expensive and the wifi kept dropping",
    "We will definitely come back to this hotel",
]
print()
print("Predictions on invented sentences:")
for sentence, topics in zip(examples, predict_topics(examples)):
    print(f"  {sentence}")
    print(f"    -> {topics}")

# Save everything needed to use the model later (vectorizer, 21 models, thresholds)
MODELS_FOLDER.mkdir(exist_ok=True)
joblib.dump(
    {"topics": TOPICS, "vectorizer": vectorizer, "models": models, "thresholds": thresholds},
    MODELS_FOLDER / "topic_model.joblib",
)
print()
print("Model saved to models/topic_model.joblib")
