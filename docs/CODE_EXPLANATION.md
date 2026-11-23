# Code Explanation: Roomor, script by script

This guide explains **all the code of the repository** so that either of us can walk the professor through any line. It follows the order in which the product works: first the three models, then the shared functions, the topic table, the report, the checks on unseen data and finally the experiments.

**State of the code: 23 November 2026.** Numbers in this guide come from our own runs (recomputed on Flavio's PC on 23 November with the versions in `requirements.txt`). DistilBERT is an experiment and is **not** part of the product: the report uses TF-IDF (decision of 23 November, Part 9).

---

## Contents

- [The product in one picture](#the-product-in-one-picture)
- [Map of the repository](#map-of-the-repository)
- [Part 0: ideas used everywhere](#part-0-ideas-used-everywhere)
- [Part 1: `topic_model.py` (Model 1, topics)](#part-1-topic_modelpy-model-1-topics)
- [Part 2: `sentiment_model.py` (Model 2, sentiment)](#part-2-sentiment_modelpy-model-2-sentiment)
- [Part 3: `pipeline.py` (shared functions and the fixed rule)](#part-3-pipelinepy-shared-functions-and-the-fixed-rule)
- [Part 4: `apply_topics.py` (the Model 3 table)](#part-4-apply_topicspy-the-model-3-table)
- [Part 5: `impact_full.py` (Model 3, impact on the score)](#part-5-impact_fullpy-model-3-impact-on-the-score)
- [Part 6: `report.py` (the "Fix first" report)](#part-6-reportpy-the-fix-first-report)
- [Part 7: checks on hotels never seen](#part-7-checks-on-hotels-never-seen)
- [Part 8: experiments and sensitivity analyses](#part-8-experiments-and-sensitivity-analyses)
- [Part 9: DistilBERT (notebooks and comparison)](#part-9-distilbert-notebooks-and-comparison)
- [Part 10: things in the code to know before the presentation](#part-10-things-in-the-code-to-know-before-the-presentation)
- [Part 11: questions the professor may ask](#part-11-questions-the-professor-may-ask)
- [Glossary](#glossary)

---

## The product in one picture

```
                 ┌──────────── trained ONCE on public data ────────────┐
                 │  HRAST (23,113 sentences)   →  Model 1 (topics)      │
                 │                             →  Model 2 (sentiment)   │
                 │  515K Booking reviews       →  Model 3 (impact)      │
                 └──────────────────────────────────────────────────────┘

 reviews of ONE hotel
   │
   ├─ Booking format (liked / disliked already separate) ─────────────┐
   │                                                                   │
   └─ one text per review (Google, TripAdvisor)                        │
        → fixed rule (split into pieces) + Model 2 (side of each piece)│
                                                                       ▼
                 cleaning ("No Negative" → empty) → windows of 12 words
                 → Model 1: which topics are criticized / praised
                 → Model 3 weights: how many score points each topic is worth
                 → report: "Fix first" (problems clearly above the average hotel)
                           "Protect"  (strengths worth the most points)
                           + 2 real sentences as examples
```

The models are never retrained for a new hotel: a hotel's reviews only go **through** them.

## Map of the repository

| File | Part | What it does | Reads | Writes |
|---|---|---|---|---|
| `src/topic_model.py` | 1 | Model 1: which of 21 topics a sentence mentions | `data/HRAST.csv` | `models/topic_model.joblib` |
| `src/sentiment_model.py` | 2 | Model 2: positive or negative, with a probability | `data/HRAST.csv` | `models/sentiment_model.joblib` |
| `src/pipeline.py` | 3 | Shared functions (cleaning, windows, topics, fixed rule). Imported, never run | `models/*.joblib` | – |
| `src/apply_topics.py` | 4 | Applies Model 1 to the 515K: one 0/1 row per review | `data/Hotel_Reviews.csv` | `data/review_topics.csv` |
| `src/impact_full.py` | 5 | Model 3: linear regression + bootstrap by hotel | `data/review_topics.csv` | `models/impact_weights.csv`, `models/impact_intercept.json` |
| `src/report.py` | 6 | The report for one hotel (`make_report`, `make_report_from_full_texts`) | `models/` | – (prints) |
| `src/verify_hotels.py` | 7.1 | Predicted vs true score of the 224 test hotels | table + weights | `results/hotel_scores.png` |
| `src/verify_full_texts.py` | 7.2 | Check of the fixed rule on the 224 test hotels and on HRAST | 515K, HRAST, `models/` | `results/full_texts_check.csv` |
| `src/check_booking_hotel.py` | 7.3 | Fixed rule on one real hotel (citizenM Zürich, Booking) | `data/citizenm_zurich_booking.csv` | – (prints) |
| `src/explain_hotel.py` | 7.4 | What Model 3 says about one hotel, review by review | `data/google_check.xlsx` | `results/citizenm_zurich_points.png` |
| `src/check_pieces.py` | 7.5 | Prepares the Excel file for the manual check | 515K, HRAST, `data/google_check.xlsx` | `data/manual_check.xlsx` |
| `src/export_split.py` | 9 | Saves the Model 1 split for DistilBERT | `data/HRAST.csv` | `data/hrast_{train,val,test}.csv` |
| `experiments/choose_thresholds.py` | 8.1 | Chooses the settings of the fixed rule on train hotels | 515K, `models/` | – (prints) |
| `experiments/impact_nothing.py` | 8.2 | Sensitivity analysis of the "nothing" rule | the two topic tables | – (prints) |
| `experiments/impact_model.py` | 8.3 | Mid-term Model 3 on 300 reviews | `data/WWRSF_HotelReviewRegression.xlsx` | – (prints) |
| `experiments/compare_distilbert.py` | 9 | Model 1 (TF-IDF) vs DistilBERT, same test sentences | `data/hrast_*.csv`, DistilBERT probabilities | – (prints) |
| `notebooks/distilbert_topics*.ipynb` | 9 | Fine-tuning of DistilBERT on Google Colab (GPU) | `data/hrast_*.csv` | probabilities + model on Drive |

**Order to run them** (from the repository root, e.g. `python src/topic_model.py`): 1 `topic_model` → 2 `sentiment_model` → 3 `apply_topics` (needs 1 and 2) → 4 `impact_full` (needs 3) → 5 `report` / checks. The trained models are already on GitHub, so to produce a report only the 515K dataset is needed.

---

## Part 0: ideas used everywhere

**Supervised learning.** We have an input `X` (what the model may use: a sentence, or 42 topic columns) and a target `y` (what it must predict: a topic, a sentiment, a score). The model learns the relation from examples where `y` is known.

**Classification vs regression.** If `y` is a category (positive / negative, topic yes / no) it is *classification* (Models 1 and 2). If `y` is a number (the score from 2.5 to 10) it is *regression* (Model 3).

**Multi-label classification.** One sentence can have several topics at once ("clean room, rude staff" → Clean, Room, Staff). Model 1 is therefore 21 separate yes/no classifiers, one per topic, not one classifier that picks one topic.

**Baseline.** A trivial method used as a minimum reference (Week 4). Every model has one: "always Room" and hand-written keywords for Model 1, "always positive" for Model 2, "always the mean score" for Model 3. A model that does not beat its baseline is not worth its complexity.

**Train / validation / test.** The model learns on *train*. Every choice we make (a threshold, a setting, the best epoch) is made on *validation*. The *test* set is used only at the end, to measure. If the test set influences a choice, its score becomes optimistic (*data leakage*). We declare it every time this happened (DistilBERT, Part 9).

**Split by hotel.** For Model 3 the split is not by review but by hotel (`GroupShuffleSplit`): all the reviews of a test hotel are in the test set and none in train. A client hotel is a hotel the model has never seen, so this is the honest test. Reviews of the same hotel resemble each other: with a split by review, the model would have already "met" every test hotel.

**Reproducibility.** Wherever randomness appears we fix the seed (`random_state=42`, `np.random.RandomState(42)`), so every run gives the same numbers.

**Paths.** Every script starts with:
```python
ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT / "data"
MODELS_FOLDER = ROOT / "models"
```
`__file__` is the path of the script itself; `.parents[1]` is the folder one level above `src/` (the repository root). So the scripts find `data/` and `models/` from wherever they are launched.

**Saving a model (`joblib`).** `joblib.dump(object, file)` writes a Python object (the vectorizer, the trained models, the thresholds) to a file; `joblib.load(file)` reads it back identical. The `.joblib` files must be loaded with the same scikit-learn version (1.8.0).

**TF-IDF.** Turns a text into numbers: one number per word of the vocabulary. TF (term frequency) = how often the word is in this text; IDF (inverse document frequency) = how rare the word is in all the texts. Common words ("the") weigh little, informative words more. `fit` learns the vocabulary and the IDF **only on train**; `transform` applies them to other texts.

**Logistic regression.** Gives each word a weight, sums weight × TF-IDF value over the words of the text, and turns the sum into a probability between 0 and 1 with the sigmoid function. Despite the name it is a classifier. Its weights can be read word by word: this is why we chose it (Week 3, "Choosing a method").

**Metrics.**
- *Accuracy*: share of correct answers.
- *Precision*: of the "yes" given by the model, how many are right. *Recall*: of the real "yes", how many the model finds. *F1*: harmonic mean of the two.
- *Micro F1*: all the 21 × N decisions pooled together; frequent topics weigh more. *Macro F1*: F1 of each topic, then the simple mean; every topic weighs the same, so rare topics matter.
- *MAE* (mean absolute error): average distance between true and predicted score, in score points. *R²*: share of the variability explained, compared with predicting the mean (1 perfect, 0 like the mean, negative worse than the mean).

**Bootstrap.** We have one sample, so one value per coefficient. To see how much it would change with other data, we draw a new sample of the same size *with replacement* from ours, refit, and repeat many times. The central 95% of the values is the interval. If it contains zero, the data cannot tell the sign of the effect: we call the coefficient **not reliable**.

**Association, not causation.** Model 3 sees that reviews criticizing a topic have lower scores on average. It does not prove that fixing that topic raises the score (common causes, missing variables, observational data).

---

## Part 1: `topic_model.py` (Model 1, topics)

**Question:** which of 21 hotel topics does a sentence talk about?
**Method:** TF-IDF + 21 logistic regressions (one per topic), one threshold per topic chosen on validation.
**Result on test:** micro F1 **0.857**, macro F1 **0.852** (keywords 0.808 / 0.807; "always Room" 0.167 / 0.017).

### Keyword lists (the baseline written by hand)

```python
KEYWORDS = {
    "Clean": ["clean", "cleaned", "cleaning", ..., "mould", "mold"],
    ...
}
```
A dictionary: for each topic, a list of words. A keyword must match a **whole** word, so "bed" does not match "beds" (plurals are listed). Multi-word keywords ("check in", "thin walls") are allowed. These lists are the second baseline: a model that does not beat them would be useless.

```python
TOPICS = ["Clean", "Comfort", "Facilities/Amenities", ..., "Generic"]
```
The 21 topic columns of HRAST. The order matters: it is the same in every script and in DistilBERT.

### Step 1: load and clean

```python
df = pd.read_csv(DATA_FOLDER / "HRAST.csv", encoding="utf-8-sig")
df = df.drop(columns=["Unnamed: 2"])
```
`utf-8-sig` removes the BOM (an invisible character at the start of the file; without it the first column would be called `﻿id`). `Unnamed: 2` is an empty column without a name.

```python
for topic in TOPICS:
    df[topic] = pd.to_numeric(df[topic], errors="coerce").fillna(0).astype(int)
```
Turns each topic column into integers. `errors="coerce"` turns anything that is not a number into "missing" (NaN), and `fillna(0)` turns it into 0. This handles the "σ" found in the Location column (row id 4634).

```python
df = df.drop_duplicates(subset="review")
```
Removes the 8 duplicate sentences (23,113 → **23,105**). A sentence in both train and test would make the test score look better than it is.

```python
n_topics = df[TOPICS].sum(axis=1)
```
`axis=1` sums **along the row**: the number of topics of each sentence. The script prints how many sentences have 0 topics and 2 or more, and the frequency of each topic (Room 5,004, Location 4,729, Staff 3,804 are the most frequent).

### Step 2: train / validation / test (70 / 15 / 15)

```python
X_rest, X_test, Y_rest, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_rest, Y_rest, test_size=0.15 / 0.85, random_state=42)
```
Two cuts. The first puts 15% of all sentences in test. The second takes, from the remaining 85%, a share of 0.15 / 0.85 ≈ 17.6%, which is 15% **of the total**. Result: **16,173 / 3,466 / 3,466**. `Y` is a table with 21 columns (one 0/1 per topic). No `stratify` here: with 21 columns at once it is not possible in a simple way.

### Step 3: the two baselines, on the test set

```python
def evaluate(name, Y_true, Y_pred):
    f1_per_topic = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
```
`average=None` gives one F1 per topic; `"micro"` and `"macro"` the two summaries (Part 0). `zero_division=0`: if a topic is never predicted, its F1 counts as 0 instead of raising a warning.

```python
Y_pred_dummy = np.zeros(Y_test.shape, dtype=int)
Y_pred_dummy[:, TOPICS.index("Room")] = 1
```
Dummy baseline: a table of zeros as big as the test set, with 1 in the whole Room column (the most frequent topic). Result 0.167 / 0.017: macro is almost zero because 20 topics out of 21 get F1 = 0.

```python
def has_keyword(sentence, words):
    for word in words:
        if re.search(r"\b" + re.escape(word) + r"\b", sentence):
            return True
```
Keyword baseline. `\b` is a *word boundary*: `\bbar\b` matches "the bar is" but not "barely". `re.escape` protects special characters such as "-" in "check-in". The double loop fills a 0/1 table: topic `j` of sentence `i` is 1 if the sentence contains one of the topic's keywords. Result **0.808 / 0.807**.

### Step 4: TF-IDF + one logistic regression per topic

```python
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)
```
One vocabulary, learned on train only, shared by the 21 models.

```python
for j, topic in enumerate(TOPICS):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, Y_train[topic])
    models[topic] = model
    P_val[:, j] = model.predict_proba(X_val_tfidf)[:, 1]
    P_test[:, j] = model.predict_proba(X_test_tfidf)[:, 1]
```
Each model learns only its own column ("does this sentence talk about Staff?"). `predict_proba` returns two columns, P(no) and P(yes); `[:, 1]` keeps P(yes). The probabilities are saved in two tables (`P_val`, `P_test`), one column per topic, so the thresholds can be chosen afterwards without retraining.

With the usual threshold 0.5 the model is **worse than the keywords** (0.797 / 0.727). Reason: most topics are rare (a few % of the sentences), so the model is rarely confident enough to pass 0.5 and misses many real mentions (low recall).

### Step 5: one threshold per topic, chosen on validation

```python
candidates = np.round(np.arange(1, 19) * 0.05, 2)      # 0.05, 0.10, ..., 0.90
for j, topic in enumerate(TOPICS):
    best_f1 = -1
    for t in candidates:
        f1 = f1_score(Y_val[topic], (P_val[:, j] >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds[j] = t
```
For each topic, 18 thresholds are tried on the **validation** set and the one with the highest F1 is kept (`>` and not `>=`: in a tie the lowest threshold wins). `np.round` avoids numbers like 0.15000000000000002. Almost all chosen thresholds are between 0.05 and 0.30: lowering the threshold lets the model say "yes" more often, which raises recall on rare topics.

```python
Y_pred_tuned = (P_test >= best_thresholds).astype(int)
```
NumPy compares each column with its own threshold (*broadcasting*). Only now the test set is used: **0.857 / 0.852**. The model beats the keywords on 15 topics of 21, above all Generic (0.16 → 0.62), Facilities (0.59 → 0.79) and Comfort (0.71 → 0.82); the keywords win on Clean, View, Beach, Bed, Lift and Value for money.

**Discovery about Generic:** its top words show that in HRAST Generic means the building and the hotel as a whole (building, design, decor, entrance, atmosphere), not vague judgements.

### Step 6: top words, prediction function, saving

```python
weights = models[topic].coef_[0]
order = weights.argsort()[::-1]
```
`coef_[0]` = one weight per vocabulary word; `argsort()` gives the positions from the lowest to the highest weight; `[::-1]` reverses them. The first 10 are the words that most push towards "yes" for Staff, Clean, Noise and Generic.

```python
thresholds = {topic: float(best_thresholds[j]) for j, topic in enumerate(TOPICS)}
def predict_topics(texts): ...
```
`predict_topics` is the function the rest of the product uses: for each text, the list of topics whose probability is at least the topic's threshold. It is tried on 3 invented sentences. The same function exists in `pipeline.py`.

```python
joblib.dump({"topics": TOPICS, "vectorizer": vectorizer, "models": models, "thresholds": thresholds},
            MODELS_FOLDER / "topic_model.joblib")
```
Everything needed to use the model later goes into one file (1.2 MB): topic order, vocabulary, 21 models, 21 thresholds.

### Limits of Model 1

- Bag of words: word order is ignored ("not as clean as expected" still contains "clean"). Model 1 finds the **topic**, not the judgement.
- Trained on single English sentences from 42 hotels in 4 cities; the 515K and client reviews are longer texts (handled with windows, Part 3).
- Rare topics (Restaurant 53 test sentences, Lift 43) have few examples, so their F1 is less stable.

---

## Part 2: `sentiment_model.py` (Model 2, sentiment)

**Question:** is a sentence positive or negative?
**Method:** TF-IDF + logistic regression (binary classification).
**Result on test:** accuracy **0.9458**, F1 0.9457 (baseline "always positive": accuracy 0.508).
**Where it is used:** not in the Model 3 table (the 515K fields are already split into negative and positive). It is used (1) by the fixed rule to send each piece of a one-text review to a side, (2) by the report to choose the most negative / positive examples, (3) by the optional "nothing" rule.

### Step 1: load and count

```python
df = pd.read_csv(DATA_FOLDER / "HRAST.csv", encoding="utf-8-sig")
print("Positive sentences:", df["positive"].sum())
```
Summing a 0/1 column counts the ones: 11,819 positive, 10,504 negative, 794 neutral. They add up to 23,117, 4 more than the 23,113 rows: 4 sentences carry two labels.

### Step 2: cleaning (four filters)

```python
df = df.drop_duplicates(subset="review")                    # Filter 1: duplicates
n_labels = df["positive"] + df["negative"] + df["neutral"]
df = df[n_labels == 1]                                      # Filter 2: exactly one label
df = df[df["neutral"] == 0]                                 # Filter 3: no neutral
df = df[df["Aspect"] == 0]                                  # Filter 4: one topic only
df = df.copy()
df["label"] = df["positive"]
```
- Filter 1: same reason as in Model 1.
- Filter 2: a sentence with two labels has no single right answer.
- Filter 3: neutral sentences are 3% and mix facts with opposite judgements; we want two classes.
- Filter 4: `Aspect = 1` marks sentences that talk about **more than one topic** (we found on 23 November that it does *not* mean "praise and criticism together": most have one sentiment, e.g. "Great location and clean rooms"). Keeping only `Aspect = 0` gives the model the simplest sentences; longer texts are cut into pieces by the fixed rule before reaching Model 2 (Part 3).
- `.copy()` makes an independent table (avoids a pandas warning); `label` is 1 for positive, 0 for negative.

Result: **19,682 sentences, 50.81% positive** (almost balanced).

### Step 3: train / test split and baseline

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
```
85 / 15: **16,729 / 2,953** sentences. `stratify=y` keeps the same share of positives in both parts.

```python
baseline = DummyClassifier(strategy="most_frequent")
```
Always answers the most frequent train class (positive): accuracy 0.5083, F1 0.674. The F1 looks decent only because a model that always says "positive" never misses a positive; accuracy is the honest number here (a coin flip).

**Why there is no validation set (decision of 23 November).** A validation set is needed only when something is *chosen*. This script chooses nothing: the threshold is the usual 0.5 and `LogisticRegression` keeps its default settings, so a validation set would never be used, and the test set is used once, only to measure. The settings that *are* chosen around Model 2 (the two thresholds of the fixed rule and whether the contrast word is removed) are chosen on 150 **train** hotels of the 515K, which act as its validation set (Part 8.1), never on the test hotels.

### Step 4: TF-IDF + logistic regression

```python
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
```
Same logic as Model 1, but one model. `max_iter=1000` gives the optimizer enough steps to converge.

| Model | Accuracy | F1 |
|---|---|---|
| Baseline (always positive) | 0.5083 | 0.6740 |
| TF-IDF + LogReg | 0.9458 | 0.9457 |

```python
print(confusion_matrix(y_test, y_pred))
```
Rows = truth, columns = prediction (0 negative, 1 positive): `[[1401, 51], [109, 1392]]`. 2,793 of 2,953 correct. 51 negatives called positive, 109 positives called negative.

```python
words = vectorizer.get_feature_names_out()
weights = model.coef_[0]
order = weights.argsort()
```
The 10 most negative words: *not, no, wasn, small, poor, bit, didn, noisy, bad, dirty*. The 10 most positive: *good, nice, great, excellent, location, clean, comfortable, amazing, friendly, spacious*. `wasn` and `didn` come from "wasn't" / "didn't": the vectorizer splits at the apostrophe and drops one-letter pieces.

Note: "location" and "clean" are among the most positive words. This is why Model 2 cannot be used to filter the negative Booking field (it would call "location was far" praise): tried and rejected on 17 November.

```python
joblib.dump({"vectorizer": vectorizer, "model": model}, MODELS_FOLDER / "sentiment_model.joblib")
```

### Limits of Model 2

- Bag of words: negation is a weak spot ("didn't hear a thing" is read as negative because of "didn't").
- It has never seen neutral sentences or sentences with several topics.
- 94.6% is on HRAST sentences only. On real reviews cut into pieces it is lower (Part 7: 92.8% on citizenM pieces).

---

## Part 3: `pipeline.py` (shared functions and the fixed rule)

This file **only defines functions and loads the saved models**: importing it does not read datasets, print or save anything. All the other scripts import from it (`from pipeline import ...`), so the same cleaning is used to build the Model 3 table, to make the report and to run the checks.

### Lists of "no criticism" texts

In the 515K, about 30% of the negative fields say "nothing to criticize" in many ways ("No Negative" 127,890 times, "Nothing", "N A", ...). If they stayed, Model 1 would find topics in them (e.g. Generic in "nothing"), and those reviews would count as criticisms.

```python
EMPTY_NEGATIVE = ["", "no negative", "nothing", "n a", ..., "leaving"]
```
Rule 1: if the **whole** negative text, in lowercase, is one of these, it becomes empty. ("leaving" is a frequent Booking placeholder.)

```python
EMPTY_NEGATIVE_EXPRESSIONS = ["can t fault", "no complaints", "liked everything", ...]
MAX_WORDS_FOR_EXPRESSIONS = 8
EXCEPTION_WORDS = ["but", "except", "apart", "only", "however", "although", "besides"]
```
Rule 2: a negative text becomes empty if it **contains** one of these expressions, but only if it has at most 8 words and none of the exception words. "can t fault anything" → empty; "can t fault the staff but the room was noisy" → kept (it has "but" and a real criticism).

```python
EMPTY_POSITIVE = ["", "no positive", "nothing", "none", "n a", "na", "no", ...]
```
Same idea for the positive field (only rule 1).

### Loading the models

```python
saved = joblib.load(MODELS_FOLDER / "topic_model.joblib")
TOPICS, vectorizer, models, thresholds = saved["topics"], saved["vectorizer"], saved["models"], saved["thresholds"]
saved_sentiment = joblib.load(MODELS_FOLDER / "sentiment_model.joblib")
```
Both models are loaded once, when the file is imported. This is why `models/` must contain the two `.joblib` files (they are on GitHub).

### Cleaning functions

```python
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\W_]", " ", text)
    return " ".join(text.split())
```
Brings any text to the format of the 515K, which has no punctuation: `[\W_]` = any character that is not a letter or a number (accented letters are kept), replaced by a space; `" ".join(text.split())` removes double spaces. "Can't fault it." → "Can t fault it". A missing value (`NaN`, which is not a string) counts as empty. This is why the lists above contain "can t" and "n a".

```python
def remove_placeholder(text, empty_list):          # rule 1
def remove_short_expression(text, expressions, max_words=8):   # rule 2
```
Rule 2 checks, in order: more than 8 words → keep; one word is an exception word → keep; an expression appears as whole words (`\b ... \b`) → empty.

```python
def remove_nothing_praise(text, threshold=0.90):
```
Optional "nothing" rule (**switched off**): if a negative text starts with "nothing" and Model 2 gives the whole text P(positive) ≥ 0.90 ("Nothing everything was perfect the staff..."), it becomes empty. Its effect was measured and found negligible (Part 8.2), so it stays off.

```python
def clean_negative(text, use_nothing_rule=False):   # normalize + rule 1 + rule 2 (+ "nothing" rule)
def clean_positive(text):                            # normalize + rule 1
```
The two functions that every script uses on the Booking fields.

### Windows of 12 words

```python
def make_windows(text, size=12, step=6):
    words = text.split()
    ...
    while True:
        windows.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
```
Model 1 learned from **single sentences**, but a 515K field can be 100 words long and has no punctuation to split it into sentences. So the text is cut into windows of 12 words that overlap by 6 (words 1–12, 7–18, 13–24, ...). The overlap makes sure that a topic sitting on the border of two windows still appears whole in one of them. A text shorter than 12 words is one window.

```python
def topics_of_text(text):
    found = set()
    for window_topics in predict_topics(make_windows(text)):
        found.update(window_topics)
    return [topic for topic in TOPICS if topic in found]
```
A text has a topic if **at least one** of its windows has it. A `set` keeps each topic once; the final list keeps the order of `TOPICS`.

```python
def topics_of_many_texts(texts):
    all_windows, owner = [], []
    for i, text in enumerate(texts):
        for window in make_windows(text):
            all_windows.append(window)
            owner.append(i)
    ...
    for k, window_topics in enumerate(predict_topics(all_windows)):
        found[owner[k]].update(window_topics)
```
Same result, much faster: the windows of **all** texts go to Model 1 in one call, and `owner[k]` remembers which text window `k` comes from. This is what makes the full 515K run in about 4 minutes.

### The fixed rule: reviews written as one text

Booking splits a review into "liked" and "disliked". Google and TripAdvisor do not: a review is one text with praise and criticism mixed. The fixed rule rebuilds a negative text and a positive text, and after that the report works exactly as with Booking.

```python
CONTRAST_WORDS = ["but", "however", "although", "though", "except", "whereas", "unfortunately"]
MIN_PIECE_WORDS = 2
MIN_COMMA_PART_WORDS = 2
LOW_POSITIVE = 0.50
HIGH_POSITIVE = 0.50
STRIP_CONTRAST_WORD = True
```
- Contrast words usually turn the judgement around: the text is cut **before** them.
- A piece shorter than 2 words is glued back to the previous one.
- Model 2 decides the side: P(positive) ≥ `HIGH` → positive, ≤ `LOW` → negative, in between → put aside. With 0.50 / 0.50, chosen on 150 train hotels (Part 8.1), **nothing is put aside** (the October idea of putting uncertain pieces aside lost more criticisms than it saved). A piece with exactly 0.50 goes to the positive side, because that test comes first.
- `STRIP_CONTRAST_WORD = True`: Model 2 reads the piece without the contrast word at its start ("but very clean" → "very clean"), because it learned from whole sentences, not from pieces starting with "but". The piece itself keeps the word.

```python
def glue_short_pieces(pieces, min_words):
```
Glues every piece shorter than `min_words` to the one before it; if the first piece is short, it is glued to the second.

```python
def split_review(text):
    for sentence in re.split(r"[.!?;\n]+", text):                     # step 1: sentences
        pattern = r"\b(?=(?:" + "|".join(CONTRAST_WORDS) + r")\b)"
        parts = [p.strip(" ,") for p in re.split(pattern, sentence, flags=re.IGNORECASE)]   # step 2
        ...
    for piece in pieces:                                               # step 3: commas
        if "," in piece and len(predict_topics([normalize_text(piece)])[0]) >= 2:
            parts = [p.strip() for p in piece.split(",") if p.strip() != ""]
```
- Step 1: cut on `.`, `!`, `?`, `;` and new lines (`[...]+` = one or more of these in a row).
- Step 2: inside each sentence, cut before each contrast word. `(?=...)` is a *lookahead*: it cuts **before** the word without removing it, so "but" stays at the start of the next piece. `\b` avoids cutting "button" at "but". `flags=re.IGNORECASE` catches "But".
- Step 3: cut on commas **only** if Model 1 finds 2 or more topics in the piece ("Great location, rude staff"). A list like "Clean, quiet and comfortable" about one topic is not cut. Model 1 runs only on pieces that contain a comma.
- The punctuation is needed here, so `split_review` runs **before** `normalize_text`.

```python
def text_for_sentiment(piece):          # removes the contrast words at the start of a piece
def route_pieces(pieces, low, high, strip_contrast):
    p_positive = sentiment_model.predict_proba(sentiment_vectorizer.transform(texts))[:, 1]
    for piece, p in zip(pieces, p_positive):
        if p >= high: positive.append(piece)
        elif p <= low: negative.append(piece)
        else: aside.append(piece)
```
Model 2 reads all the pieces in one call and each piece goes to a list.

```python
def split_full_text(text, low, high, strip_contrast):
    pieces = split_review(text)
    negative, positive, aside = route_pieces(pieces, low, high, strip_contrast)
    negative = [clean_negative(p) for p in negative]
    positive = [clean_positive(p) for p in positive]
    return negative_text, positive_text, len(pieces), len(aside)
```
One review in, a negative text and a positive text out (plus two counts for the report). Each piece is cleaned **on its own**, so "No complaints" inside a longer review is still recognized as "no criticism" by the same rules used on Booking fields.

---

## Part 4: `apply_topics.py` (the Model 3 table)

**Question:** for each of the 515K reviews, which topics are criticized and which are praised?
**Output:** `data/review_topics.csv`, 515,212 rows × 44 columns (`Hotel_Address`, `Reviewer_Score`, 21 `Neg_` and 21 `Pos_` columns of 0/1). Time: about 4 minutes.

```python
USE_NOTHING_RULE = False
```
Switch of the "nothing" rule. With `True` the script writes a **different** file (`review_topics_nothing.csv`), so the original table is never overwritten (used by `impact_nothing.py`).

### Step 1: load and clean

```python
df = pd.read_csv(DATA_FOLDER / "Hotel_Reviews.csv")
df = df.drop_duplicates()
df = df[["Hotel_Address", "Negative_Review", "Positive_Review", "Reviewer_Score"]].copy()
for col in ["Hotel_Address", "Negative_Review", "Positive_Review"]:
    df[col] = df[col].str.strip()
```
Duplicates are removed on **all** the columns (515,738 → 515,212, the 526 duplicates), and only then the 4 useful columns are kept: removing duplicates after dropping columns would also delete different reviews that happen to have the same text and score. `Hotel_Address` identifies the hotel (1,493 addresses for 1,492 names). `.str.strip()` removes spaces at the borders (" No Negative " → "No Negative").

The script prints the 30 most frequent texts of each field: this is how the placeholders ("No Negative", "Nothing", ...) were found.

### Steps 2–2c: exploration (why the cleaning rules exist)

- Step 2 shows the topics found in 5 random reviews, to check by eye.
- Step 2b takes 20,000 reviews and lists the negative texts whose **only** topic is Generic: most were ways of saying "nothing to criticize" ("can t fault it", "liked everything"). This is what motivated rule 2.
- Step 2c tests rule 2 on 5 sentences (only the first must become empty) and counts how many texts it empties and how the "only Generic" list shrinks.

### Step 3: `build_table`

```python
def build_table(data):
    neg_texts = [clean_negative(t, USE_NOTHING_RULE) for t in data["Negative_Review"]]
    pos_texts = [clean_positive(t) for t in data["Positive_Review"]]
    neg_topics = topics_of_many_texts(neg_texts)
    pos_topics = topics_of_many_texts(pos_texts)
    for topic in TOPICS:
        table["Neg_" + topic] = [1 if topic in topics else 0 for topics in neg_topics]
    ...
```
`Neg_Staff = 1` means: Model 1 found Staff in the **negative** field of this review. Note: it is the field that is negative, not necessarily the sentence ("staff could not have been nicer" written in the negative field still counts as a criticism; Part 10). The function is first timed on 5,000 reviews to estimate the full run. Two quick tests of the "nothing" rule follow.

### Step 4: full run and summary

```python
full_table.to_csv(output_file, index=False)
pct_neg = full_table["Neg_" + topic].mean() * 100
```
The mean of a 0/1 column is the share of ones. The script prints, for each topic, the % of reviews that criticize and praise it (e.g. Location is praised in 48.5% of the reviews), and how many reviews have no criticism or no praise at all.

---

## Part 5: `impact_full.py` (Model 3, impact on the score)

**Question:** how many score points is each topic worth, when criticized and when praised?
**Method:** multiple linear regression on 42 columns of 0/1, split by hotel, bootstrap by hotel.
**Result on hotels never seen:** MAE **0.968** vs 1.288 of the baseline; R² **0.36** vs −0.009. 38 of 42 weights reliable (37 used by the report).

### Step 1: load and look

```python
df = pd.read_csv(DATA_FOLDER / "review_topics.csv")
neg_cols = [col for col in df.columns if col.startswith("Neg_")]
```
The script prints rows, columns, number of hotels, missing values, mean (8.4) and median (8.8) of the score, and the share of reviews with no topic at all.

### Step 2: split by hotel and baseline

```python
X = df[neg_cols + pos_cols]
y = df["Reviewer_Score"]
groups = df["Hotel_Address"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
```
`GroupShuffleSplit` puts 15% of the **hotels** in test, with all their reviews. `next(...)` takes the only split produced. Result: **1,269 hotels / 432,531 reviews in train, 224 hotels / 82,681 reviews in test**, 0 hotels in both (the script checks it with `hotels_train & hotels_test`, the intersection of two sets).

```python
train_mean = y_train.mean()
y_pred_base = np.full(len(y_test), train_mean)
```
Baseline: every test review gets the mean score of train (8.372). MAE 1.288, R² −0.009 (slightly negative because the train mean is not exactly the test mean).

### Step 3: linear regression

```python
model = LinearRegression()
model.fit(X_train, y_train)
```
The model finds the equation

`score = intercept + c1·Neg_Clean + ... + c21·Neg_Generic + c22·Pos_Clean + ... + c42·Pos_Generic`

with the numbers that minimise the squared error on train. Since every column is 0/1, a coefficient is simply the points added or removed when that topic is criticized / praised, **holding the other 41 columns fixed**.

- **Intercept 8.338:** predicted score of a review with no topic at all.
- **Largest weights:** criticism of Staff −0.80, Clean −0.70, Room −0.61, Bed −0.50, Bathroom −0.46, Noise −0.42; praise of Staff +0.68, Generic +0.47, View +0.28.
- **Asymmetry (like the Kano model):** Clean −0.70 when criticized, only +0.08 when praised; Bathroom −0.46 / +0.10; Value for money −0.39 / +0.06. View the opposite: −0.01 / +0.28. Some things matter only when they go wrong, others only when they go well. This is why criticized and praised topics are separate columns.
- Location is praised in 48.5% of the reviews but is worth only +0.17.

```python
coefs = pd.Series(model.coef_, index=X_train.columns)
share = X_train.mean() * 100
```
`share` = % of train reviews where each column is 1; it becomes `share_pct`, the "average hotel" used by the report.

### Step 4: bootstrap by hotel

```python
rows_by_hotel = list(pd.Series(range(len(groups_train))).groupby(groups_train).indices.values())
for _ in range(200):
    drawn = rng.randint(0, n_hotels, size=n_hotels)
    idx = np.concatenate([rows_by_hotel[h] for h in drawn])
    boot_model = LinearRegression()
    boot_model.fit(X_train_arr[idx], y_train_arr[idx])
```
- `rows_by_hotel`: for each train hotel, the positions of all its reviews.
- Each resample draws 1,269 **hotels** with replacement and takes all their reviews (twice if a hotel is drawn twice).
- **Why by hotel and not by review:** reviews of the same hotel are not independent (same building, same staff). Resampling single reviews pretends we have 432,531 independent pieces of information and gives intervals that are too narrow: by hotel they are on average 1.66 times wider (3.4 times for Pos_Staff).
- `rng` is created once with seed 42, outside the loop. Data are converted to plain NumPy arrays because taking rows from them is much faster. 200 resamples take about 8 minutes.

```python
lower = boot_coefs.quantile(0.025)
upper = boot_coefs.quantile(0.975)
unreliable = (lower <= 0) & (upper >= 0)
```
**4 weights not reliable:** Neg_Beach, Neg_View, Pos_Parking, Pos_Beach (rare topics in city hotels). Neg_Bar is just reliable with 200 resamples (−0.060, −0.001) but touches zero with 1,000, so the report excludes it by hand: **37 weights used**. The main weights are very precise (Neg_Staff from −0.83 to −0.77). Pos_Lift (−0.21), Pos_Reception (−0.07) and Pos_Wi-Fi (−0.03) are reliable but negative: the report never shows them as strengths.

### Step 5: comparison with the mid-term model and saving

The 10 coefficients of the 300-review model (Part 8.3) are compared with the full model: same sign 10 out of 10, 9 out of 10 inside the earlier interval. The full weights are closer to zero (Neg_Room −1.10 → −0.61) because with 42 columns the effect of the room is shared among Room, Bed, Bathroom, Clean and Comfort, and because errors of Model 1 pull weights towards zero (Part 10).

```python
weights.to_csv(MODELS_FOLDER / "impact_weights.csv", index_label="topic")
json.dump({"intercept": float(model.intercept_)}, f)
```
`impact_weights.csv`: one row per column with `topic, coef, lower, upper, share_pct`. The intercept goes to a separate JSON file so the CSV keeps exactly one row per topic column.

---

## Part 6: `report.py` (the "Fix first" report)

**Question for the manager:** what should I fix first, and what should I protect?
**Main functions:** `make_report(negative_texts, positive_texts)` (Booking format) and `make_report_from_full_texts(texts)` (one text per review). `print_report` prints the result. Importing the file runs nothing; `python src/report.py` runs a test on 3 hotels.

### Settings

```python
N_TOP = 3          # Fix first topics with examples; Protect topics shown
N_EXAMPLES = 2     # real sentences per topic
MIN_WORDS = 4      # minimum words in an example
MIN_REVIEWS = 30   # below this: warning "results are indicative only"
MIN_GAP = 0.05     # Fix first only if the hotel is at least 5 percentage points above the average...
MIN_CRITICS = 5    # ...and at least 5 reviews criticize the topic
EXCLUDED_WEIGHTS = ["Neg_Bar"]
EXCLUDED_FROM_FIX_FIRST = ["Generic"]
READABLE_NAMES = {"Clean": "Cleanliness", "Reception-checkin": "Check-in & reception", ...}
```
Generic is excluded from Fix first because in a manual check 13 of the 19 criticisms counted as Generic were wrong (it stays in the regression and can appear in Protect). `READABLE_NAMES` turns the HRAST column names into names a hotel manager understands.

### `load_weights`

```python
contains_zero = (table["lower"] <= 0) & (table["upper"] >= 0)
weights = table.loc[~contains_zero & ~excluded, "coef"].to_dict()
averages = table["share_pct"].to_dict()
```
`~` means "not". Returns the reliable weights only, and the average shares of **all** columns.

### `make_report`: the logic

1. The two texts of each review are cleaned (`clean_negative`, `clean_positive`) and Model 1 finds their topics, exactly as when the Model 3 table was built.
2. **Fix first.** For each topic (except Generic):
   - `share` = share of the hotel's reviews that criticize it;
   - `average` = the same share in the average train hotel (`share_pct / 100`);
   - `gap = share − average`;
   - the topic enters only if its `Neg_` weight is reliable and negative, `gap ≥ 0.05` and at least 5 reviews criticize it;
   - `extra_points = gap × weight` = points lost **compared with the average hotel**; `points = share × weight` = all the points lost.
   - Topics are sorted by `extra_points` (most negative first). Examples are added for the first 3.
   - If no topic qualifies: "No topic is clearly worse than the average hotel".
3. **Protect.** For each topic with a reliable **positive** `Pos_` weight: `points = share × weight`; the 3 topics with the most points are shown.
4. Warning if the hotel has fewer than 30 reviews.

**Why compare with the average hotel?** With the absolute points (`share × weight`) Staff and Room were at the top in all the hotels tried, even in the best one: almost every hotel gets some staff and room criticism. A manager needs what is **unusually** bad in his hotel.

### `find_examples`

```python
for i, (text, topics) in enumerate(zip(texts, topics_per_text)):
    if topic not in topics: continue
    for window in make_windows(text):
        if len(window.split()) >= MIN_WORDS: ...
window_topics = predict_topics(windows)
p_positive = sentiment_model.predict_proba(sentiment_vectorizer.transform(windows))[:, 1]
candidates.sort(key=lambda c: c[0], reverse=most_positive)
```
Takes the windows (at least 4 words) of the reviews that contain the topic, keeps those where Model 1 really finds the topic, and sorts them by Model 2's P(positive): the most negative first for Fix first, the most positive first for Protect. At most one window per review, and a window already shown for another topic is never repeated (`used_windows`). **Model 2 only chooses the examples; it never changes the counts.** So the examples can look more precise than the numbers behind them.

### Confidence of each Fix first topic (`negative_sounding_reviews`, added on 23 November)

```python
SOUNDS_NEGATIVE = 0.50
LOW_CONFIDENCE = 0.50
sounding = negative_sounding_reviews(neg_texts, neg_topics, [row["topic"] for row in fix_first])
row["n_negative"] = len(sounding[row["topic"]])
row["confidence"] = row["n_negative"] / row["n_critics"]
row["check"] = row["confidence"] < LOW_CONFIDENCE
```
Model 1 finds the topic, not the judgement, so praise written in the negative field is counted as a criticism. For each Fix first topic the report therefore also says **how many of the reviews that criticize it sound really negative**: a review counts if at least one of its windows contains the topic (Model 1) and Model 2 gives it P(positive) below 0.50. All the windows go to the two models in one call. If fewer than half sound negative, the report prints "only X of Y reviews sound negative: some may be praise, check the examples before deciding", and that topic gets examples even if it is not in the first 3 (the slide of Week 8: "show the confidence, not just the answer").

It **never changes the counts or the ranking**: the weights of Model 3 were learned without this filter, so filtering the counts would mix two different measures.

What we saw on 23 November (our run, not yet checked against the manual labels):
- Savoy, Mövenpick, Casa Mimosa: the report is identical to before; the new column shows mostly high values (e.g. Savoy Room 120 of 121, Staff 46 of 54; Mövenpick Location 69 of 94). No topic is flagged.
- Over all the criticisms counted (not only Fix first), Casa Mimosa has 83% that sound negative, Mövenpick 90%: the direction is right (Casa Mimosa is the hotel with the most wrong counts) but the size is not (the manual check estimated about 47% wrong at Casa Mimosa). So the confidence catches only part of the errors.
- Model 2 misses praise written with a negation: "the staff could not have been nicer" gets P(positive) 0.01. Plain praise ("Staff were lovely", 0.98) is caught.
- For reviews written as one text (citizenM), the column is always full (e.g. 11 of 11): the pieces were already sent to the negative side by Model 2, so it adds nothing there.

### `make_report_from_full_texts`

```python
for text in texts:
    negative, positive, pieces, aside = split_full_text(text)
```
Each one-text review goes through the fixed rule (Part 3), and then `make_report` does the rest. The report also says how many pieces were read and how many were put aside (with 0.50 / 0.50: zero).

### Test on 3 hotels (`if __name__ == "__main__":`)

The same split by hotel as `impact_full.py` is rebuilt, and three test hotels are chosen: the lowest mean score, the one closest to the median, the highest. Results:

| Hotel | True mean | Fix first |
|---|---|---|
| Savoy Amsterdam | 6.01 | 9 topics, first Room |
| Mövenpick Amsterdam | 8.51 | Value for money, Location, Check-in |
| H10 Casa Mimosa | 9.66 | No topic clearly worse than the average |

---

## Part 7: checks on hotels never seen

### 7.1 `verify_hotels.py`: predicted vs true score of the 224 test hotels

```python
shares = by_hotel[topic_cols].mean()
"predicted": intercept + (shares * weights).sum(axis=1)
```
For each test hotel: share of its reviews in each of the 42 columns × weight, summed, plus the intercept. Because the regression is linear, this is exactly the mean of the predictions of its reviews. **All 42 weights** are used here (we check the whole model, not the report).

| | Model 3 | Baseline (always 8.37) |
|---|---|---|
| MAE by hotel | **0.310** | 0.523 |
| Correlation predicted / true | **0.907** | – |

- The improvement is 41% by hotel, 25% by review (0.968 vs 1.288): averaging over many reviews cancels part of the error. The fair comparison is always with the baseline at the same level.
- **Compression:** the predicted scores are less spread than the true ones (standard deviation 0.299 vs 0.641). The worst hotels are predicted too high (+0.37 on average for the lowest third), the best too low (−0.41). Worst case: Savoy, 7.50 predicted vs 6.01 true. **The ranking of hotels is reliable, the size is not:** the points in the report are relative priorities, not points that can be won back.
- Probable cause (not proven): *attenuation bias*. Errors in the 0/1 columns of Model 1 push the weights towards zero, so predictions shrink towards the mean. Also, the model knows whether a topic is criticized, not how strongly.
- Does not prove: that each topic is right (errors can cancel out), that fixing a problem raises the score, that it works for Swiss hotels or other languages. Topics and score come from the same reviews: it is not a forecast.

The chart (`results/hotel_scores.png`) puts one dot per hotel with the 45° line "predicted = true", same limits on both axes so the line is really at 45°.

### 7.2 `verify_full_texts.py`: does the fixed rule work?

**Idea:** Booking reviews already contain the right answer (which text is liked and which disliked). We glue the two fields into one text, give it to the fixed rule + Model 2, and compare the result with the two original fields. Only the 224 test hotels, looked at **once**, after the settings were chosen on train hotels (Part 8.1).

```python
def glue(neg, pos, separator, negative_first):
```
Glues the two cleaned fields (the order is random for each review) and remembers, for each word, which field it came from. Two versions: **A** with a full stop between the fields (like a real review), **B** with only a space (the hardest case: the 515K has no punctuation, only contrast words can cut).

Three methods are compared:
- **All negative** (baseline): the whole text goes to the negative side (what would happen without the rule);
- **Model 2, no split** (baseline): the whole text is one piece, Model 2 picks its side;
- **Fixed rule + Model 2**: `split_full_text`.

Metrics, each against the two Booking fields: F1 of the 0/1 topic tables (criticized and praised), MAE and correlation of the predicted hotel score, share of hotels with the **same #1 in Fix first**, and *Jaccard overlap* of the Fix first and Protect lists (topics in common / topics in either list).

| Version A (full stop) | Same #1 in Fix first | F1 criticized | F1 praised | Correlation |
|---|---|---|---|---|
| All negative | 5.4% | – | – | – |
| Model 2, no split | 40.6% | – | – | – |
| **Fixed rule + Model 2** | **81.7%** | **0.921** | **0.951** | 0.896 (Booking fields 0.907) |

Without punctuation (B) the rule helps little (same #1 in 50%). The only number that does not improve is the MAE by hotel (0.314 vs 0.301 of "Model 2, no split"), but that method gets the #1 wrong in 59% of hotels.

`piece_sides` checks the side of each piece: the true side is the field of most of its words ("mixed" if words of both). The script then checks on **HRAST with real punctuation**: the test sentences of Model 2 glued 4 at a time into invented reviews → 92.6% of the sentences on the right side with the rule, 67.9% without. Finally the `Aspect = 1` sentences: of the 864 with a contrast word, the rule cuts 97.5% and separates praise and criticism in 59.8%. Saves `results/full_texts_check.csv`.

### 7.3 `check_booking_hotel.py`: one real Swiss hotel

```
python src/check_booking_hotel.py                        (data/citizenm_zurich_booking.csv)
python src/check_booking_hotel.py data/other_hotel.csv
```
Input: a CSV with `score, liked, disliked, text_if_unknown, fields_known`. 30 Booking reviews of citizenM Zürich (July–November 2026) copied by hand, with **real punctuation**; 21 have the two blocks distinguishable (`fields_known = yes`), the others are used only in the full report.

- Step 1: every piece of every block is sent to a side; the right side is the block it comes from. **91.7%** right. The pieces on the "wrong" side are printed to check by hand (is the rule wrong, or did the guest write praise in "disliked"?).
- Step 2: same three methods as 7.2, glued with a new line and with a space. F1 of topics 0.925 / 0.930, and **Fix first (Room, Noise, Facilities, Comfort) and Protect (Staff, Hotel overall, Location) identical** to the ones built from the Booking blocks.
- Step 3: the report from all 30 reviews, each as one text: what the demo would show.

### 7.4 `explain_hotel.py`: what Model 3 says about one hotel

```
python src/explain_hotel.py                                  (data/google_check.xlsx)
python src/explain_hotel.py data/other.xlsx "Hotel name"
```
Each review (one text, columns `text` and `score`) goes through the fixed rule, Model 1 and Model 3. For each review: guest's score, predicted score, topics criticized and praised with their weights.

- **Hotel level:** predicted mean **8.17** vs true **8.33** (error 0.16; average on the 224 hotels 0.31).
- **Review level:** MAE 0.64 vs 0.58 of the baseline "always 8.372", correlation 0.48. **Model 3 is not good for single reviews**: only the aggregated hotel numbers are shown to the manager.
- Points by topic (reliable weights only, same as the report): lost −0.85 (Room −0.22, Cleanliness −0.09, Bed −0.08, Noise −0.08), gained +0.69 (Staff +0.25).
- Chart `results/citizenm_zurich_points.png`: one bar per topic and side, warm color = lost, cool = gained, only topics worth at least 0.01 points.

### 7.5 `check_pieces.py`: the manual check

Prepares `data/manual_check.xlsx` with 3 sheets; the columns in CAPITALS are empty and filled by hand.
- `google_pieces`: every piece of the citizenM reviews, with side chosen and P(positive);
- `wrong_side`: 30 pieces of the 515K (3,000 reviews of **train** hotels) that ended on the "wrong" side;
- `contrast_sentences`: 30 HRAST sentences with `Aspect = 1` and a contrast word, with the pieces and their sides.

`read_reviews` accepts Excel or CSV (it tries both `;` and `,` as separators, because Excel in Italian saves CSV with `;`). The labels were put by Claude and are summarized in `results/manual_check_summary.md` (**to be reviewed by us** before using them): citizenM 93.2% of cuts right, 92.8% of pieces on the right side, **all 46 criticisms on the negative side**; every error is praise or a neutral fact sent to negative (about 21% of the negative pieces, e.g. "didn't hear a thing from inside our room" counted as Noise). Of the 30 "wrong side" pieces of the 515K, only 47% are real errors of the rule (37% are guests writing in the wrong Booking field).

---

## Part 8: experiments and sensitivity analyses

These scripts are not part of the product: they justify choices.

### 8.1 `experiments/choose_thresholds.py`: settings of the fixed rule

```python
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
```
The script lives in `experiments/`, so it tells Python to look for `pipeline.py` in `src/`.

- **Validation = 150 train hotels** drawn at random (never the test hotels). Same glue as version A.
- **Settings tried:** 5 pairs of thresholds (0.30/0.70 ... 0.50/0.50) × contrast word stripped or not = 10.
- **Rule decided before looking:** criterion = Fix first overlap with the Booking fields; a setting replaces the current one (0.35 / 0.65, word kept) only if it is better by more than 0.01.
- **Result:** 0.50 / 0.50 with the word stripped, overlap **0.850** vs 0.817 → adopted (now in `pipeline.py`). Stripping the word matters little (0.850 vs 0.847).

### 8.2 `experiments/impact_nothing.py`: does the "nothing" rule matter?

Builds Model 3 on the two tables (rule off / on), with the same split by hotel. Checks first that the tables differ only because of the rule (same rows, identical `Pos_` columns, every changed review has no `Neg_` topic left). Result: the rule changes **582 reviews out of 515,212 (0.11%)**, largest change of a weight 0.006 (Neg_Generic, −0.453 → −0.460), MAE 0.9685 vs 0.9681. **Decision: the rule stays off.**

### 8.3 `experiments/impact_model.py`: the mid-term Model 3 (300 reviews)

The first version of Model 3, presented at the mid-term on 20 October. Data: `WWRSF_HotelReviewRegression.xlsx`, 300 random reviews of the 515K (seed 42) with 10 columns (5 topics × criticized / praised) found **with keywords**, not with Model 1.

- **5-fold cross-validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`): with 300 rows a single test set of 45–60 reviews would be too noisy (the baseline MAE across folds goes from 1.16 to 1.43). Each review is used once for testing.
- Baseline `DummyRegressor(strategy="mean")`: MAE 1.282, R² −0.017. Linear regression: **MAE 1.011, R² 0.333**, better in all 5 folds.
- Final fit on all 300 to read the coefficients (intercept 8.33): Neg_Room −1.10, Neg_Staff −1.01, Neg_Location −0.88, Pos_Staff +0.89, Pos_Restaurant +0.41.
- Bootstrap by review (1,000 resamples): 5 of 10 coefficients not reliable (rare topics: Noise, Neg_Location barely reliable).
- Limits: only 300 reviews, keyword topics (they make mistakes: "no towels left after service" counted as a Staff criticism), only 5 topics.

Superseded by `impact_full.py`; kept to show how the project grew.

---

## Part 9: DistilBERT (notebooks and comparison)

**Question:** can a pretrained neural network recognize the 21 topics better than TF-IDF?
**Answer on HRAST: yes.** 8 epochs, three seeds: test micro F1 **0.890** / macro F1 **0.880** (mean), all three seeds beat 0.857 / 0.852.
**Decision (23 November): it stays an experiment and is not part of the product.** `report.py` uses TF-IDF. Reasons:
1. It passed only the first of the two criteria decided on 22 November (beat TF-IDF on HRAST sentences). The second (beat MAE 0.97 on hotels never seen, i.e. improve the real product) was never tested; keeping it anyway would mean changing the rule after seeing the results.
2. It does not fix the largest error we measured: Model 1 says what a piece talks about, not whether it is a criticism (praise written in the "disliked" field still counts as criticism).
3. Connecting it means redoing the whole chain: DistilBERT on the 515K reviews split into windows (hours of GPU), a new Model 3, new checks, every number changes.
4. 268 MB and very slow without a GPU: the pipeline would no longer run on a normal laptop from the README, and a free online demo could crash or take minutes.

Not tried yet (optional): DistilBERT as **Model 2** (sentiment). It would target a real error of the fixed rule (negations such as "didn't hear a thing") without redoing the Model 3 table, because Model 2 is used only by the fixed rule and to choose the report's examples.

### `src/export_split.py`

Repeats Steps 1 and 2 of `topic_model.py` and saves `data/hrast_train.csv`, `hrast_val.csv`, `hrast_test.csv`, so DistilBERT is trained and tested on **exactly** the same sentences. Several `assert` stop the script if anything differs: sizes 16,173 / 3,466 / 3,466, column order, sentences and topics re-read from the files, no sentence in two files. `keep_default_na=False`: a sentence like "None" must stay a text.

### The notebooks (Google Colab, Tesla T4 GPU)

| Notebook | What it does |
|---|---|
| `distilbert_topics.ipynb` | First training, 3 epochs, seed 42 |
| `distilbert_topics_8epochs.ipynb` | 8 epochs, keeps the epoch with the best macro F1 **on validation** |
| `distilbert_topics_seeds.ipynb` | The same 8-epoch training with seed 1 and seed 2 (stability check) |

Each starts with `QUICK_TEST = True` (300 / 100 / 100 sentences, 1–2 epochs, files named `QUICKTEST_...`): only to check that every cell works, its numbers mean nothing.

**Key ideas, cell by cell:**
- **Settings** fixed before seeing results: `distilbert-base-uncased`, learning rate 2e-5, batch 32, maximum 64 tokens, seed 42. `fix_seed()` fixes Python, NumPy and PyTorch randomness.
- **Tokenizer:** a model reads numbers, not letters. The tokenizer splits a sentence into *tokens* (rare words into several pieces) and adds `[CLS]` and `[SEP]`. *Uncased* = everything in lowercase. It must be the same tokenizer used in pre-training.
- **Batches:** `padding` makes all the sentences of a batch the same length; the *attention mask* tells the model which tokens are padding. Only train is shuffled; validation and test keep their order so each probability stays next to its sentence.
- **Pre-training and fine-tuning (Week 5):** DistilBERT already learned English from Wikipedia and books (a smaller, faster BERT). It knows that "staff" and "personnel" are used alike and it reads word order, which TF-IDF cannot. We add a new final layer with 21 outputs (random at first: hence the "newly initialized" warning) and adjust all the weights a little on our sentences.
- **Sigmoid, not softmax:** softmax makes the 21 probabilities sum to 1 (the topics compete: one right answer). Sigmoid turns each score into a probability on its own, so a sentence can have several topics. `problem_type="multi_label_classification"` uses *binary cross-entropy*, one per topic: like 21 yes/no questions, the same structure as Model 1.
- **Training loop (Week 6):** for each batch the model computes the error (*loss*); `loss.backward()` computes how to move each weight (backpropagation); `clip_grad_norm_` limits too-large corrections; `optimizer.step()` moves the weights (AdamW, weight decay 0.01, a form of regularization); `scheduler.step()` lowers the learning rate linearly from 2e-5 to 0; `zero_grad()` clears the computations. `model.train()` switches dropout on, `model.eval()` off; `torch.no_grad()` computes without learning.
- **Best epoch (8-epoch notebooks):** at the end of each epoch, thresholds are chosen on validation and micro / macro F1 computed on validation; if the macro F1 is the highest so far, a copy of the weights is kept, and at the end the model goes back to the best epoch. The test is never used here. Training loss falling while validation loss rises would mean the model is memorizing (overfitting): this is why the best epoch is kept (Lecture 7, early stopping).
- **Outputs:** the model saved on Google Drive (reloaded and checked: same predictions), and CSV files with the probabilities on validation and test (`distilbert8_val_probs.csv`, ...), to put in `data/` (not on GitHub).

### `experiments/compare_distilbert.py`

```python
thresholds = choose_thresholds(P_val)        # same rule as topic_model.py, on validation
pred_bert = (P_test >= thresholds).astype(int)
```
DistilBERT's thresholds are chosen on validation with the same rule as Model 1, so the comparison is fair. Then, on the test set: F1 per topic, micro and macro F1, and the errors (topics added by mistake, topics missed, sentences completely right).

```python
def micro_macro(Y_true, Y_pred):
    tp = ((Y_true == 1) & (Y_pred == 1)).sum(axis=0)
    ...
for _ in range(1000):
    rows = rng.randint(0, len(Y_test), len(Y_test))
    differences.append([b[0] - a[0], b[1] - a[1]])
```
F1 is computed from the counts (true positives, false positives, false negatives) because it is much faster, which the bootstrap needs. **Paired bootstrap:** 1,000 times, the same resampled test sentences are given to both models, and the difference is recomputed. If the 95% interval of the difference does not contain zero, the difference is larger than chance (in the test sentences).

**Seed rule (decided before the trainings with seed 1 and 2):** DistilBERT is kept only if **all three seeds** beat 0.857 / 0.852 on the test.

| Seed | Test micro F1 | Test macro F1 |
|---|---|---|
| 42 (epoch 7) | 0.8898 | 0.8774 |
| 1 (epoch 7) | 0.8908 | 0.8780 |
| 2 (epoch 8) | 0.8901 | 0.8832 |
| TF-IDF (Model 1) | 0.8567 | 0.8517 |

For seed 42: difference +0.033 micro (95% interval +0.027 to +0.039), +0.026 macro (+0.014 to +0.036); topics added by mistake 879 → 608, missed 741 → 621, sentences completely right 63.8% → 72.9%. Better on 20 topics of 21 (Generic 0.625 → 0.736); slightly worse on the rarest (Restaurant, 53 test sentences; Lift, 43). The model to use is **seed 42**: choosing seed 2 because its test score is highest would be exactly the mistake to avoid.

**To declare (Trustworthy AI):** the test was computed **four times** (3 epochs, 8 epochs, seed 1, seed 2). The decision to train longer was taken after seeing the first test result, so the 8-epoch result may be slightly optimistic; the last two runs only applied a rule written before. All runs use the same data split.

---

## Part 10: things in the code to know before the presentation

- **Model 1 counts topics, not judgements.** In the negative field, a sentence that praises ("staff could not have been nicer") is counted as a criticism. Measured on 50 reviews of two extreme hotels: about 30% of the counts wrong (12% at the Savoy, 47% at Casa Mimosa). Labels put by Claude, **to be reviewed by us**.
- **Praise read as criticism in one-text reviews.** On citizenM about 21% of the pieces sent to the negative side were praise or neutral facts; no criticism was lost. So the report does not miss problems but can inflate one (Noise with "didn't hear a thing").
- **Weights are compressed and are associations** (Part 7.1): use them to rank, not as points to win back, and never as causes.
- **Single-review scores are not reliable** (citizenM: worse than the baseline); only hotel-level numbers are shown.
- **Only English, needs punctuation and enough reviews:** non-English reviews are ignored without a warning; with 30 reviews the #1 problem matches the full report in about 53% of the cases, with 200 in about 80% (quick test by Claude, not in the repository).
- **Topics that do not exist among the 21:** heat / air conditioning complaints are spread over Facilities and Comfort.
- **`explain_hotel.py` writes the train mean (8.372) by hand**, as printed by `impact_full.py`: if Model 3 is trained again on other data, it must be updated.

---

## Part 11: questions the professor may ask

**Data and splits**
1. *Why two datasets?* HRAST has topic and sentiment labels per sentence but no score; the 515K has the score but no topic labels. Models 1 and 2 learn to read sentences on HRAST, Model 3 links topics to the score on the 515K.
2. *Why split Model 3 by hotel?* A client is a hotel never seen. Reviews of one hotel are similar; a split by review would let the model meet every test hotel during training.
3. *Why a validation set?* Every choice (thresholds, rule settings, best epoch) is made on it, so the test score stays honest.
4. *Why does Model 2 have no validation set?* It does not choose anything (fixed 0.5 threshold, default settings), so a validation set would never be used; the test is used once. The thresholds of the fixed rule, which are chosen, were chosen on 150 train hotels.
5. *Where did you use the test set more than once?* DistilBERT, four times; declared. Everything else once.

**Model 1**
6. *Why 21 logistic regressions and not one model?* Multi-label: a sentence can have several topics, so each topic is its own yes/no question.
7. *Why one threshold per topic?* With 0.5, rare topics were almost never predicted (worse than keywords). Thresholds chosen on validation raise recall: 0.797 → 0.857 micro F1.
8. *Micro vs macro F1?* Micro pools all decisions (frequent topics dominate); macro averages topics (rare topics count as much as frequent ones).
9. *Why windows of 12 words?* Model 1 learned from sentences; the 515K fields are long and have no punctuation. Windows imitate sentences; the overlap avoids cutting a topic in two.

**Model 2 and the fixed rule**
10. *Why not use Model 2 to clean the negative Booking field?* Tried: it threw away more real criticisms than praise, because it learned topic words like "location" and "clean" as positive.
11. *How does the fixed rule work?* Cut on punctuation, before contrast words, and on commas when a piece has two topics; Model 2 sends each piece to a side.
12. *How do you know it works?* Booking reviews have the right answer: we glued the two fields and compared. Same #1 problem in 81.7% of the 224 test hotels (baselines 41% and 5%); on a real Swiss hotel, report identical to the Booking blocks.
13. *Why 0.50 / 0.50?* Chosen on 150 train hotels with a rule written before: putting uncertain pieces aside lost more criticisms than it saved.

**Model 3 and the report**
14. *Why linear regression?* The product is the coefficients: points per topic, readable by a manager.
15. *Why separate columns for criticized and praised?* Some topics matter mostly when bad (cleanliness), others when good (view).
16. *What does −0.80 for Neg_Staff mean?* Reviews that criticize the staff have on average a score 0.80 points lower, holding the other topics fixed. Association, not cause.
17. *Why bootstrap by hotel?* Reviews of the same hotel are not independent; by review the intervals would be too narrow.
18. *Why rank against the average hotel?* With absolute points Staff and Room came first everywhere. The manager needs what is unusually bad in his hotel.
19. *Why are predicted hotel scores compressed?* Probably attenuation bias from Model 1's errors; so the report gives priorities, not points to recover.
20. *What happens if the tool is wrong and the manager renovates the wrong thing?* Every topic comes with real sentences to check; each Fix first topic says how many of its criticisms really sound negative, and asks to check the examples when fewer than half do; the report compares with the average and shows few topics; a warning appears with few reviews; the numbers are presented as priorities, not guarantees. (To be completed in Trustworthy AI.)

**DistilBERT**
21. *Why sigmoid and not softmax?* Several topics can be true together.
22. *What is fine-tuning?* Starting from a model pretrained on general English and adjusting its weights on our task: far less data needed than training from scratch.
23. *Why is it not in the product?* Better on HRAST sentences, but it never passed the second criterion (hotels never seen), it does not fix the main error (topic, not judgement), it would require rebuilding the whole chain, and it is 268 MB and slow without a GPU. Next step: prove it improves the report.
24. *Why three seeds?* To check that the gain is not luck. The rule (all three must beat TF-IDF) was written before the trainings.

## Glossary

| Term | Meaning |
|---|---|
| DataFrame | A table in pandas |
| Feature / target | Input variable (a column of `X`) / variable to predict (`y`) |
| Multi-label | One example can have several labels at once |
| Baseline | Trivial method every model must beat |
| Train / validation / test | Learn / choose / measure |
| Data leakage | Information from the test set entering training or choices |
| Split by hotel (`GroupShuffleSplit`) | All the reviews of a hotel on the same side of the split |
| TF-IDF | Text → numbers: frequency in the text × rarity in all texts |
| Logistic regression | Weighted sum of features → probability (sigmoid); a classifier |
| Threshold | Probability above which the answer is "yes" |
| Accuracy | Share of correct predictions |
| Precision / recall | Right among the predicted "yes" / found among the real "yes" |
| F1, micro, macro | Harmonic mean of precision and recall; pooled / averaged by topic |
| Confusion matrix | Table of true vs predicted classes |
| Linear regression | Score = intercept + sum of coefficient × column |
| Intercept | Prediction when all columns are 0 |
| Coefficient (weight) | Points added or removed when a column is 1 |
| MAE | Mean absolute error, in score points |
| R² | Share of variability explained, compared with the mean |
| Cross-validation (k-fold) | Each part of the data is the test set once |
| Bootstrap | Resampling with replacement to measure uncertainty |
| Paired bootstrap | Same resampled examples given to two models, to compare them |
| Reliable weight | 95% bootstrap interval that does not contain zero |
| Attenuation bias | Errors in the inputs push coefficients towards zero |
| Jaccard overlap | Items in common / items in either list |
| Confidence (report) | Share of the reviews criticizing a topic that Model 2 reads as negative |
| Window | 12 consecutive words of a text (overlapping by 6) |
| Fixed rule | Hand-written splitting of a one-text review into pieces |
| Contrast word | "but", "however", ...: the text is cut before it |
| Token / tokenizer | Piece of a word / tool that turns text into numbered pieces |
| Pre-training / fine-tuning | Learning general language / adapting to our task |
| Epoch | One full pass over the training data |
| Learning rate | Size of each weight correction |
| Sigmoid / softmax | Independent probabilities / probabilities that sum to 1 |
| Binary cross-entropy | Error of a yes/no prediction given as a probability |
| Overfitting / early stopping | Memorizing train / keeping the epoch best on validation |
| Dropout, weight decay | Regularization: switch off neurons at random, shrink weights |
| `joblib` | Saves and loads Python objects (the trained models) |
