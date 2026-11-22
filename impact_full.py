import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# ---------- Step 1: load the Model 3 table and look at it ----------

# Load the table built by apply_topics.py (one row per review)
df = pd.read_csv("Data/review_topics.csv")

# The 21 Neg_ columns and the 21 Pos_ columns, found by their name
neg_cols = [col for col in df.columns if col.startswith("Neg_")]
pos_cols = [col for col in df.columns if col.startswith("Pos_")]

print("Rows:", df.shape[0], "- Columns:", df.shape[1])
print("Neg_ columns:", len(neg_cols), "- Pos_ columns:", len(pos_cols))
print("Different hotels:", df["Hotel_Address"].nunique())

# Missing values: how many in each column (we print only the columns that have some)
missing = df.isna().sum()
print("Missing values in total:", missing.sum())
if missing.sum() > 0:
    print(missing[missing > 0])

print("Mean score:", round(df["Reviewer_Score"].mean(), 3))
print("Median score:", df["Reviewer_Score"].median())

# Reviews with no topic at all (0 criticized AND 0 praised)
n_topics = df[neg_cols + pos_cols].sum(axis=1)
print("Reviews with no topic (neither criticized nor praised):", (n_topics == 0).sum(),
      f"({(n_topics == 0).mean() * 100:.1f}%)")

# ---------- Step 2: split BY HOTEL and baseline ----------

# X = the 42 topic columns (input), y = the score (what we want to predict)
X = df[neg_cols + pos_cols]
y = df["Reviewer_Score"]
groups = df["Hotel_Address"]          # the "group" of each review is its hotel

# GroupShuffleSplit: 15% of the HOTELS go to the test set, with ALL their reviews
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))   # row positions of train and test

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
hotels_train = set(groups.iloc[train_idx])
hotels_test = set(groups.iloc[test_idx])

print()
print("Train: hotels", len(hotels_train), "- reviews", len(train_idx),
      f"({len(train_idx) / len(df) * 100:.1f}%)")
print("Test:  hotels", len(hotels_test), "- reviews", len(test_idx),
      f"({len(test_idx) / len(df) * 100:.1f}%)")
print("Hotels in BOTH train and test:", len(hotels_train & hotels_test))

# Baseline: predict for every test review the mean score of the TRAIN set
train_mean = y_train.mean()
y_pred_base = np.full(len(y_test), train_mean)      # an array with the same number repeated

print()
print("Train mean score:", round(train_mean, 3))
print("Baseline MAE:", round(mean_absolute_error(y_test, y_pred_base), 4))
print("Baseline R2: ", round(r2_score(y_test, y_pred_base), 4))

# ---------- Step 3: multiple linear regression on the 42 topic columns ----------

model = LinearRegression()
model.fit(X_train, y_train)               # learns one coefficient per column + the intercept
y_pred = model.predict(X_test)            # predicted scores for the reviews of unseen hotels

# Comparison table on the test set
print()
print(f"{'Model':<22}{'MAE':>10}{'R2':>10}")
print(f"{'Baseline (mean)':<22}{mean_absolute_error(y_test, y_pred_base):>10.4f}"
      f"{r2_score(y_test, y_pred_base):>10.4f}")
print(f"{'Linear regression':<22}{mean_absolute_error(y_test, y_pred):>10.4f}"
      f"{r2_score(y_test, y_pred):>10.4f}")

# Intercept and coefficients
coefs = pd.Series(model.coef_, index=X_train.columns)   # one coefficient per column, with its name
share = X_train.mean() * 100                            # % of train reviews where the column is 1

print()
print("Intercept:", round(model.intercept_, 2))

print()
print("Neg_ coefficients (from the most negative):")
for col in coefs[neg_cols].sort_values().index:
    print(f"  {col:<32}{coefs[col]:>7.2f}   ({share[col]:.1f}% of reviews)")

print()
print("Pos_ coefficients (from the most positive):")
for col in coefs[pos_cols].sort_values(ascending=False).index:
    print(f"  {col:<32}{coefs[col]:>7.2f}   ({share[col]:.1f}% of reviews)")
