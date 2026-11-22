# Code Explanation: Study Guide for the Two Models

This guide explains the two scripts in this repository so that either of us can walk the professor through them line by line.

| Script | Question it answers | Method |
|---|---|---|
| `sentiment_model.py` | Is a hotel-review sentence **positive or negative**? | TF-IDF + logistic regression (classification) |
| `impact_model.py` | **How many score points** does a review gain or lose when it mentions a topic? | Multiple linear regression + bootstrap (regression) |

How to run (from the repository root folder, with the datasets inside `Data/`):

```
python src/sentiment_model.py
python experiments/impact_model.py
```

The datasets are **not** on GitHub (`Data/` is in `.gitignore`), so they have to be shared outside GitHub.

---

## Part 0: Ideas used by both scripts

**Features and target.** In supervised learning we have an input `X` (the information the model may use) and a target `y` (the thing to predict). The model learns the relation between them from examples where `y` is known.

**Classification vs regression.** If `y` is a category (positive / negative) it is *classification*. If `y` is a number (a score from 2.5 to 10) it is *regression*.

**Baseline.** A trivial method used as a minimum reference. If a model does not beat the baseline, its complexity buys nothing.

**Train and test.** The model learns on the *train* data and is judged on *test* data it has never seen. The test set must never be used while training or tuning, otherwise the score is optimistic and says nothing about new data (this is called *data leakage*).

**Reproducibility.** Wherever randomness appears we fix `random_state=42` so that every run gives the same numbers.

---

## Part 1: `sentiment_model.py` (sentiment classification)

### Data

`Data/HRAST.csv` has 23,113 sentences from hotel reviews. Important columns:

- `review`: the sentence.
- `positive`, `negative`, `neutral`: 0/1 flags with the sentiment label.
- `Aspect`: 0 for a simple sentence, 1 for a mixed one (2,884 sentences).
- Many other 0/1 columns for hotel topics (Clean, Staff, ...), which we do not use here.
- One column with an empty name, which pandas calls `Unnamed: 2`. We ignore it.

### Step 1: load and inspect

```python
df = pd.read_csv("Data/HRAST.csv", encoding="utf-8-sig")
```
Reads the CSV into a table (`DataFrame`). `utf-8-sig` tells Python to discard the BOM, an invisible character at the start of the file; without it the first column would be called `﻿id`.

```python
print("Number of rows:", len(df))
print("Columns:", list(df.columns))
print("Positive sentences:", df["positive"].sum())
```
`len(df)` counts rows, `df.columns` lists the column names, and summing a 0/1 column counts the ones. Output: 23,113 rows, 11,819 positive, 10,504 negative, 794 neutral.

The three counts add up to 23,117, which is 4 more than the number of rows. So 4 sentences carry two labels. This is why a cleaning step is needed.

### Step 2: cleaning

Each filter removes data that would confuse or mislead the model.

```python
df = df.drop_duplicates(subset="review")
```
**Filter 1, duplicates.** Removes rows with the same text (8 of them), keeping the first. If the same sentence is in both train and test, the model has already "seen the answer" and the test score looks better than it really is.

```python
n_labels = df["positive"] + df["negative"] + df["neutral"]
df = df[n_labels == 1]
```
**Filter 2, exactly one label.** Sums the three flags per row; a total of 1 means one label. `df[condition]` keeps only the rows where the condition is true. Rows with two labels are ambiguous: there is no single correct answer to learn.

```python
df = df[df["neutral"] == 0]
```
**Filter 3, drop neutral.** We want a two-class problem (positive vs negative). After Filter 2, every remaining row has exactly one label, so a row with `neutral == 0` is either positive or negative.

```python
df = df[df["Aspect"] == 0]
```
**Filter 4, simple sentences only.** Mixed sentences (`Aspect == 1`) contain both praise and criticism, so one label does not describe them well. We will handle them later with a separate rule.

```python
df = df.copy()
df["label"] = df["positive"]
```
`.copy()` makes an independent table (it avoids a pandas warning when adding a column). `label` is 1 for positive and 0 for negative; since neutral rows are gone, `positive` already is that label.

Result: **19,682 sentences, 50.81% positive**, so the classes are almost balanced.

### Step 3: train/test split and baseline

```python
X = df["review"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
```
- `X` is the text, `y` the correct label.
- `test_size=0.15`: 15% of the sentences go to the test set (2,953), 85% to train (16,729).
- `random_state=42`: fixes the shuffle, so the split is reproducible.
- `stratify=y`: keeps the same share of positive sentences (about 50.8%) in train and test.

```python
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
```
The baseline ignores the text and always answers with the most frequent train class (positive). `fit` here only counts which class is most common.

Baseline results: **accuracy 0.5083, F1 0.674**.
Accuracy is the share of correct predictions; F1 is the harmonic mean of precision and recall for the positive class. The F1 looks decent only because the dummy says "positive" every time, so it never misses a positive. Accuracy is the honest number here: about a coin flip.

### Step 4: TF-IDF + logistic regression

```python
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```
**TF-IDF** turns text into numbers. Each word in a sentence gets a score combining:
- **TF (term frequency):** how often the word appears in that sentence.
- **IDF (inverse document frequency):** how rare the word is across all sentences. Very common words ("the") get a low weight, informative rare words a higher one.

Each sentence becomes a long list of numbers, one per vocabulary word.

`fit_transform` on the train set learns the vocabulary and the IDF values from the train sentences and converts them. `transform` on the test set only applies what was learned. If we called `fit` on the test set, test information would leak into the model.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
```
**Logistic regression** gives each word a weight, multiplies the weights by the sentence's TF-IDF scores, and sums everything. The sum is turned into a probability between 0 and 1 by the sigmoid function; above 0.5 the sentence is predicted positive, otherwise negative. Training searches for the weights that make the fewest mistakes on the train set. Despite the name, it is a classifier. `max_iter=1000` allows enough iterations to converge.

```python
print(f"{'Model':<22}{'Accuracy':>10}{'F1':>10}")
```
Prints the comparison table. `:<22` left-aligns in 22 characters and `:>10.4f` right-aligns a number with 4 decimals.

| Model | Accuracy | F1 |
|---|---|---|
| Baseline | 0.5083 | 0.6740 |
| TF-IDF + LogReg | 0.9458 | 0.9457 |

```python
print(confusion_matrix(y_test, y_pred))
```
Rows are the truth, columns the prediction (order: 0 = negative, 1 = positive).

| | Predicted negative | Predicted positive |
|---|---|---|
| **True negative** | 1401 | 51 |
| **True positive** | 109 | 1392 |

The diagonal (1401 + 1392 = 2793 of 2953, i.e. 94.6%) are correct predictions. There are 51 negatives called positive and 109 positives called negative: the model errs about twice as often on positives.

```python
words = vectorizer.get_feature_names_out()
weights = model.coef_[0]
order = weights.argsort()
```
`get_feature_names_out()` lists the vocabulary, `coef_[0]` gives the weight of each word, and `argsort()` returns the indices sorted from lowest to highest weight. The first 10 are the most negative words, the last 10 the most positive.

How to read a weight: a positive weight pushes the sentence towards "positive", a negative one towards "negative", and a larger absolute value means a stronger push.
- Most negative: *not, no, wasn, small, poor, bit, didn, noisy, bad, dirty*.
- Most positive: *good, nice, great, excellent, location, clean, comfortable, amazing, friendly, spacious*.

These make sense. Note that `wasn` and `didn` come from "wasn't" and "didn't": the vectorizer splits at the apostrophe and drops one-letter pieces like "t".

### Limits to mention for `sentiment_model.py`

- The model looks at each word on its own (a "bag of words"), so negation such as "not bad" is a weak spot.
- We removed neutral and mixed sentences, so the model has only seen clear-cut cases.
- It was evaluated on one train/test split; with about 3,000 test sentences that is reasonably stable.
- Accuracy 94.6% is on this dataset only; hotel reviews from other sources could behave differently.

---

## Part 2: `impact_model.py` (impact of topics on the score)

### Data

`Data/WWRSF_HotelReviewRegression.xlsx`, sheet `Recensioni`: 300 real Booking.com reviews.

- `Reviewer_Score`: the score given, from 2.5 to 10 (mean 8.347). This is `y`.
- Five topics: Staff, Room, Noise, Location, Restaurant. For each there are two 0/1 columns, `Neg_<Topic>` (the negative text mentions the topic) and `Pos_<Topic>` (the positive text mentions it), found by keyword matching. That gives 10 columns, which together form `X`.

Reading the `.xlsx` file needs the `openpyxl` library.

### Step 1: load and inspect

```python
df = pd.read_excel("Data/WWRSF_HotelReviewRegression.xlsx", sheet_name="Recensioni")
topic_cols = [c for c in df.columns if c.startswith(("Neg_", "Pos_"))]
```
`read_excel` reads one sheet. The second line builds a list with the names of the columns that start with `Neg_` or `Pos_`, which are the 10 topic columns. Then a loop prints `df[col].sum()` for each, the number of reviews mentioning that topic.

| Topic | Neg (count) | Pos (count) |
|---|---|---|
| Staff | 37 | 132 |
| Room | 112 | 110 |
| Noise | 21 | 15 |
| Location | 16 | 156 |
| Restaurant | 50 | 78 |

Some topics are rare (Noise, Neg_Location). This matters later.

### Step 2: X, y, cross-validation and baseline

```python
X = df[topic_cols]
y = df["Reviewer_Score"]
```
`X` is a 300 × 10 table of 0/1 values; `y` is the score. In a regression, `X` holds what the model may use and `y` what it must predict.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```
**5-fold cross-validation:** the 300 reviews are shuffled and cut into 5 parts of 60. In turn, each part is the test set while the other four are the train set. We get 5 scores and average them.

**Why not a single split?** With 300 rows, a 15% test set would be only 45 reviews, and the score would depend heavily on which 45 landed there. The baseline's MAE across the folds ranges from 1.16 to 1.43, which shows how unstable one split would be. Cross-validation uses every review for both training and testing (never in the same fold), so the estimate is more stable and no data is wasted.

```python
baseline = DummyRegressor(strategy="mean")
```
The baseline predicts, for every test review, the mean score of the train fold. If we know nothing about a review, the average is the most sensible guess, and it uses only train data so there is no leakage. A model that uses the topics must beat it, otherwise the topics add no information.

**MAE (Mean Absolute Error):** for each review take the absolute difference between true and predicted score, then average. It has the same unit as the score: an MAE of 1.28 means the prediction is off by 1.28 points on average. Lower is better.

Baseline MAE averaged over the 5 folds: **1.2823**.

### Step 3: linear regression

```python
for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```
`kf.split(X)` gives 5 pairs of row positions (train rows, test rows). `enumerate(..., start=1)` numbers the folds from 1. `.iloc` selects rows by position. Baseline and linear regression are evaluated inside the same loop, so both see exactly the same folds.

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)
```
Multiple linear regression looks for the equation

`score = intercept + c1·Neg_Staff + c2·Neg_Room + ... + c10·Pos_Restaurant`

choosing the numbers that minimise the squared error on the train set.

**R²:** the share of the variability in the scores that the model explains, compared with the mean-only baseline. 1 means perfect, 0 means no better than the mean, and negative means worse than the mean.

| Model | MAE | R² |
|---|---|---|
| Baseline (mean) | 1.2823 | -0.0173 |
| Linear regression | 1.0107 | 0.3329 |

The regression beats the baseline in all 5 folds. The topics explain about a third of the variability; the rest depends on things the keywords do not capture.

```python
final_model = LinearRegression()
final_model.fit(X, y)
coefs = pd.Series(final_model.coef_, index=topic_cols).sort_values()
```
After the evaluation, we train once on all 300 reviews to read the coefficients. No test set is needed here because we are not measuring quality (cross-validation already did that). `pd.Series(..., index=topic_cols)` attaches names to the coefficients and `sort_values()` orders them.

**Intercept = 8.33:** the predicted score for a review that mentions none of the 10 topics (all columns equal 0). It is the starting point to which the coefficients add or subtract.

**Coefficients:**

| Topic | Coefficient |
|---|---|
| Neg_Room | -1.10 |
| Neg_Staff | -1.01 |
| Neg_Location | -0.88 |
| Neg_Restaurant | -0.34 |
| Neg_Noise | -0.22 |
| Pos_Location | 0.08 |
| Pos_Room | 0.31 |
| Pos_Noise | 0.36 |
| Pos_Restaurant | 0.41 |
| Pos_Staff | 0.89 |

**How to read a coefficient.** `Neg_Room = -1.10` means: *holding the other nine topics fixed, a review that criticises the room has on average a score 1.10 points lower than one that does not.* `Pos_Staff = +0.89` means a review praising the staff scores 0.89 points higher on average.

**Association, not causation.** The model only sees that reviews mentioning a topic have different scores on average. It does not show that the mention caused the score.
- General dissatisfaction can cause both the low score and the complaints (a common cause).
- Missing variables (price, trip type, expectations) can affect both the topics mentioned and the score.
- The data are observational: nobody changed the room while keeping everything else equal. Only an experiment could establish causation.

So we say "is associated with a score X points lower", never "makes the score drop by X".

### Step 4: bootstrap

```python
rng = np.random.RandomState(42)
for _ in range(1000):
    idx = rng.randint(0, len(df), size=len(df))
    boot_model = LinearRegression()
    boot_model.fit(X.iloc[idx], y.iloc[idx])
    boot_coefs.append(boot_model.coef_)
```
**Bootstrap idea.** We have one sample of 300 reviews, so one value per coefficient, and we do not know how much it would change with another 300 reviews. The bootstrap simulates this: draw 300 reviews *with replacement* from our sample (like drawing tickets from an urn and putting each back), refit the regression, and repeat 1000 times. If a coefficient barely moves across the 1000 refits it is stable; if it moves a lot it is uncertain.

- `rng` is created once, outside the loop, with seed 42. Re-seeding inside the loop would produce 1000 identical resamples.
- `rng.randint(0, len(df), size=len(df))` draws 300 row positions with replacement: some reviews appear several times, others not at all.

```python
lower = boot_coefs.quantile(0.025)
upper = boot_coefs.quantile(0.975)
unreliable = lower[name] <= 0 <= upper[name]
```
For each coefficient we take the 2.5th and 97.5th percentiles of its 1000 values: the central 95% range. If zero is inside the interval, the data are compatible with "no effect", and we mark the coefficient **NOT RELIABLE**. This does not mean the effect is zero; it means 300 reviews are not enough to tell.

| Topic | Coef | Lower | Upper | Verdict |
|---|---|---|---|---|
| Neg_Room | -1.10 | -1.46 | -0.76 | reliable |
| Neg_Staff | -1.01 | -1.60 | -0.52 | reliable |
| Neg_Location | -0.88 | -1.66 | -0.08 | reliable (barely) |
| Neg_Restaurant | -0.34 | -0.73 | 0.06 | not reliable |
| Neg_Noise | -0.22 | -0.94 | 0.42 | not reliable |
| Pos_Location | 0.08 | -0.23 | 0.38 | not reliable |
| Pos_Room | 0.31 | -0.02 | 0.62 | not reliable |
| Pos_Noise | 0.36 | -0.17 | 0.85 | not reliable |
| Pos_Restaurant | 0.41 | 0.11 | 0.69 | reliable |
| Pos_Staff | 0.89 | 0.61 | 1.14 | reliable |

**Why rare topics have wider intervals.** A coefficient compares the mean score of the reviews that mention the topic with the mean of the others. If a topic appears in only 15-20 reviews, that mean rests on very few scores, and a handful of unusual reviews can shift it a lot. In each bootstrap resample the number of those reviews drawn varies (sometimes 8, sometimes 25), so the coefficient jumps around and the interval widens. Compare `Neg_Noise` (21 mentions, interval width 1.36) and `Pos_Noise` (15 mentions, width 1.02) with `Neg_Room` (112 mentions, width 0.70) and `Pos_Staff` (132 mentions, width 0.53).

`Neg_Location` (16 mentions) has a wide interval but it stays below zero because the effect is very large. Its upper end (-0.08) is close to zero, and `Neg_Restaurant` (0.06) and `Pos_Room` (-0.02) are just on the other side, so these labels could change with another seed or more data.

### Limits to mention for `impact_model.py`

- Only 300 reviews, and the topics are found by keywords, so some mentions are missed and some are false hits.
- Only five topics are in the model; everything else is lumped into the unexplained part.
- Coefficients are associations, not causal effects.
- Five of the ten coefficients are not reliable, so only the other five should be interpreted.
- Linear regression assumes the effects of topics simply add up; interactions are not modelled.

---

## Part 3: Questions the professor may ask

1. **Why a baseline?** To have a minimum reference. A model that does not beat it learns nothing useful.
2. **Why must the test set stay untouched?** It simulates unseen data. Using it during training or tuning makes the score optimistic.
3. **Why `fit` on train and only `transform` on test (TF-IDF)?** Learning the vocabulary from the test set would leak test information into the model.
4. **Why `stratify=y`?** To keep the class proportions equal in train and test.
5. **Why remove duplicates?** The same sentence in train and test would inflate the score.
6. **Why cross-validation in the second model?** With 300 rows a single split is too noisy.
7. **What is the difference between MAE and R²?** MAE is the average error in score points; R² is the share of variability explained, relative to the mean.
8. **What does a coefficient mean?** The average change in score when the topic is present, holding the other topics fixed.
9. **Why is it not causal?** Common causes, omitted variables and observational data.
10. **What is the bootstrap for?** To measure how stable each coefficient is, by refitting on 1000 resamples.
11. **What does "not reliable" mean?** The 95% interval contains zero, so the data cannot tell whether the effect is positive, negative or null.
12. **Why are intervals wider for rare topics?** Few observations make the group mean unstable.
13. **What are the weak points of the sentiment model?** It treats words independently (negation) and was trained only on clear, simple sentences.

## Glossary

| Term | Meaning |
|---|---|
| DataFrame | A table in pandas |
| Feature | An input variable (a column of `X`) |
| Target | The variable to predict (`y`) |
| Overfitting | Learning the train data too closely, so performance on new data is poor |
| Data leakage | Information from the test set entering the training |
| Accuracy | Share of correct predictions |
| Precision / Recall | Of the predicted positives how many are right / of the real positives how many are found |
| F1 | Harmonic mean of precision and recall |
| Confusion matrix | Table of true vs predicted classes |
| MAE | Mean absolute error |
| R² | Share of variability explained relative to the mean |
| Coefficient | The weight of a variable in the model |
| Intercept | The prediction when all variables are 0 |
| Fold | One of the parts in cross-validation |
| Bootstrap | Resampling with replacement to measure uncertainty |
