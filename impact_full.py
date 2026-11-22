import json
import time
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

# ---------- Step 4: bootstrap BY HOTEL on the train set ----------

start = time.time()

# Random number generator with a fixed seed, so the results are reproducible
rng = np.random.RandomState(42)

# For each train hotel, the row positions (inside X_train) of all its reviews
groups_train = groups.iloc[train_idx].to_numpy()
rows_by_hotel = list(pd.Series(range(len(groups_train))).groupby(groups_train).indices.values())
n_hotels = len(rows_by_hotel)

# Plain arrays: taking rows from them is faster than from a DataFrame
X_train_arr = X_train.to_numpy(dtype=float)
y_train_arr = y_train.to_numpy()

n_boot = 200
boot_coefs = []
for _ in range(n_boot):
    # Draw as many hotels as in the train set, WITH replacement:
    # some hotels appear several times, others never
    drawn = rng.randint(0, n_hotels, size=n_hotels)
    # Take ALL the reviews of the drawn hotels (twice if the hotel was drawn twice)
    idx = np.concatenate([rows_by_hotel[h] for h in drawn])
    boot_model = LinearRegression()
    boot_model.fit(X_train_arr[idx], y_train_arr[idx])
    boot_coefs.append(boot_model.coef_)

# Table with 200 rows (one per resample) and 42 columns (one per coefficient)
boot_coefs = pd.DataFrame(boot_coefs, columns=X_train.columns)

# 95% interval: from the 2.5th to the 97.5th percentile of each coefficient
lower = boot_coefs.quantile(0.025)
upper = boot_coefs.quantile(0.975)
unreliable = (lower <= 0) & (upper >= 0)      # does the interval contain zero?

elapsed = time.time() - start

print()
print(f"Bootstrap by hotel: {n_boot} resamples of {n_hotels} hotels")
print(f"{'Topic':<32}{'Coef':>7}{'Lower':>9}{'Upper':>9}  Reliability")
neg_order = list(coefs[neg_cols].sort_values().index)
pos_order = list(coefs[pos_cols].sort_values(ascending=False).index)
for col in neg_order + pos_order:             # same order as in Step 3
    flag = "NOT RELIABLE" if unreliable[col] else "reliable"
    print(f"{col:<32}{coefs[col]:>7.2f}{lower[col]:>9.3f}{upper[col]:>9.3f}  {flag}")

print()
print("Not reliable coefficients:", unreliable.sum(), "out of", len(unreliable))
print(f"Bootstrap time: {elapsed:.1f} seconds")

# ---------- Step 5: comparison with the trial Model 3 and saving the weights ----------

# Coefficients of the trial model (impact_model.py, 300 reviews).
# There the topics were found with keywords, and Restaurant also included breakfast.
trial_coefs = {
    "Neg_Room": -1.10,
    "Neg_Staff": -1.01,
    "Neg_Location": -0.88,
    "Neg_Restaurant (dinner)": -0.34,
    "Neg_Noise": -0.22,
    "Pos_Location": 0.08,
    "Pos_Room": 0.31,
    "Pos_Noise": 0.36,
    "Pos_Restaurant (dinner)": 0.41,
    "Pos_Staff": 0.89,
}

print()
print("Comparison with the trial model (300 reviews, keyword topics):")
print(f"{'Topic':<26}{'Trial':>8}{'Full':>8}{'Full - Trial':>14}")
for col, trial in trial_coefs.items():
    print(f"{col:<26}{trial:>8.2f}{coefs[col]:>8.2f}{coefs[col] - trial:>14.2f}")
print("Note: in the trial model Restaurant also included breakfast, here it is only dinner.")

# Save one row per coefficient, for the report generator
weights = pd.DataFrame({
    "coef": coefs,
    "lower": lower,            # 2.5th percentile of the bootstrap
    "upper": upper,            # 97.5th percentile of the bootstrap
    "share_pct": share,        # % of train reviews where the column is 1
})
weights.to_csv("models/impact_weights.csv", index_label="topic")
print()
print("Saved models/impact_weights.csv with", len(weights), "rows")

# Save the intercept in a separate file: predicted score = intercept + sum of the coefficients
# of the topics in the review (the CSV keeps exactly one row per topic column)
with open("models/impact_intercept.json", "w") as f:
    json.dump({"intercept": float(model.intercept_)}, f)
print("Saved models/impact_intercept.json:", round(model.intercept_, 3))
