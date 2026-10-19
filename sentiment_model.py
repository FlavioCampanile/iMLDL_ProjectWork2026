import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset (utf-8-sig removes the BOM at the start of the file)
df = pd.read_csv("Data/HRAST.csv", encoding="utf-8-sig")

# Number of rows (= sentences) in the dataset
print("Number of rows:", len(df))

# Names of all the columns
print("Columns:", list(df.columns))

# The positive/negative/neutral columns contain 1 or 0: summing them counts the sentences
print("Positive sentences:", df["positive"].sum())
print("Negative sentences:", df["negative"].sum())
print("Neutral sentences:", df["neutral"].sum())

# ---------- Step 2: dataset cleaning ----------

# Filter 1: remove duplicate sentences (same text in "review"), keeping the first
df = df.drop_duplicates(subset="review")

# Filter 2: keep only sentences with exactly ONE sentiment label
n_labels = df["positive"] + df["negative"] + df["neutral"]
df = df[n_labels == 1]

# Filter 3: remove neutral sentences (only positive or negative remain)
df = df[df["neutral"] == 0]

# Filter 4: keep only simple sentences (Aspect = 0)
df = df[df["Aspect"] == 0]

# Create the label column: 1 = positive, 0 = negative
df = df.copy()
df["label"] = df["positive"]

print("Sentences remaining:", len(df))
print("Percentage of positive:", round(df["label"].mean() * 100, 2), "%")

# ---------- Step 3: train/test split and baseline ----------

# X = the texts (input), y = the labels (correct answer)
X = df["review"]
y = df["label"]

# Split: 85% train, 15% test; random_state fixes the randomness; stratify keeps the class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print("Train sentences:", len(X_train), "- Test sentences:", len(X_test))

# Baseline: always predicts the most frequent class in the train set
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

print("Baseline accuracy:", round(accuracy_score(y_test, y_pred_base), 4))
print("Baseline F1:", round(f1_score(y_test, y_pred_base), 4))

# ---------- Step 4: TF-IDF + logistic regression ----------

# TF-IDF: learns the vocabulary ONLY from the train set, then transforms train and test
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic regression trained on the train set
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Comparison table on the test set
acc_model = accuracy_score(y_test, y_pred)
f1_model = f1_score(y_test, y_pred)
print()
print(f"{'Model':<22}{'Accuracy':>10}{'F1':>10}")
print(f"{'Baseline':<22}{accuracy_score(y_test, y_pred_base):>10.4f}{f1_score(y_test, y_pred_base):>10.4f}")
print(f"{'TF-IDF + LogReg':<22}{acc_model:>10.4f}{f1_model:>10.4f}")

# Confusion matrix: rows = true, columns = predicted (order: 0 negative, 1 positive)
print()
print("Confusion matrix (rows = true, columns = predicted; order 0=negative, 1=positive):")
print(confusion_matrix(y_test, y_pred))

# Word weights: one weight for each word in the vocabulary
words = vectorizer.get_feature_names_out()
weights = model.coef_[0]
order = weights.argsort()  # indices from the lowest weight to the highest

print()
print("10 most NEGATIVE words:")
for i in order[:10]:
    print(f"  {words[i]:<20}{weights[i]:>8.3f}")
print("10 most POSITIVE words:")
for i in order[::-1][:10]:
    print(f"  {words[i]:<20}{weights[i]:>8.3f}")
