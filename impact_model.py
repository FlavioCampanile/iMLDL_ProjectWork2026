import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the sheet "Recensioni" from the Excel file
df = pd.read_excel("Data/WWRSF_HotelReviewRegression.xlsx", sheet_name="Recensioni")

# Number of rows (= reviews)
print("Number of reviews:", len(df))

# Average of the score given by the reviewers
print("Average Reviewer_Score:", round(df["Reviewer_Score"].mean(), 3))

# Topic columns: the ones starting with Neg_ or Pos_ (1 = the topic is mentioned, 0 = not)
topic_cols = [c for c in df.columns if c.startswith(("Neg_", "Pos_"))]

# Summing a 0/1 column counts how many times it equals 1
print("Number of reviews mentioning each topic:")
for col in topic_cols:
    print(f"  {col:<18}{df[col].sum():>5}")

# ---------- Step 2: X, y, cross-validation and baseline ----------

# X = the 10 topic columns (input), y = the score to predict (target)
X = df[topic_cols]
y = df["Reviewer_Score"]

# 5-fold cross-validation: the 300 reviews are shuffled and split into 5 parts;
# each part is used once as test set, the other 4 as train set
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# In every fold we evaluate two models on the same train/test split:
# - baseline: predicts the mean score of the train fold for every review
# - linear regression: predicts the score from the 10 topic columns
baseline_maes, baseline_r2s = [], []
lr_maes, lr_r2s = [], []
for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    baseline_maes.append(mean_absolute_error(y_test, y_pred_base))
    baseline_r2s.append(r2_score(y_test, y_pred_base))

    # Linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)
    lr_maes.append(mean_absolute_error(y_test, y_pred_lr))
    lr_r2s.append(r2_score(y_test, y_pred_lr))

    print(f"Fold {fold}: MAE baseline = {baseline_maes[-1]:.4f}, MAE linear regression = {lr_maes[-1]:.4f}")

# Comparison table: mean over the 5 folds
print()
print(f"{'Model':<20}{'MAE':>8}{'R2':>8}")
print(f"{'Baseline (mean)':<20}{sum(baseline_maes) / 5:>8.4f}{sum(baseline_r2s) / 5:>8.4f}")
print(f"{'Linear regression':<20}{sum(lr_maes) / 5:>8.4f}{sum(lr_r2s) / 5:>8.4f}")

# ---------- Step 3b: final model on all 300 reviews ----------

# Train on all the data to read the coefficients (no test set needed here: we are not predicting)
final_model = LinearRegression()
final_model.fit(X, y)

print()
print("Intercept:", round(final_model.intercept_, 2))

# One coefficient per topic column, sorted from the most negative to the most positive
coefs = pd.Series(final_model.coef_, index=topic_cols).sort_values()
print("Coefficients (sorted):")
for name, value in coefs.items():
    print(f"  {name:<18}{value:>7.2f}")

# ---------- Step 4: bootstrap confidence intervals ----------

# Random number generator with a fixed seed, so the results are reproducible
rng = np.random.RandomState(42)

n_boot = 1000
boot_coefs = []
for _ in range(n_boot):
    # Draw 300 row positions WITH replacement: some reviews appear several times, others never
    idx = rng.randint(0, len(df), size=len(df))
    boot_model = LinearRegression()
    boot_model.fit(X.iloc[idx], y.iloc[idx])
    boot_coefs.append(boot_model.coef_)

# Table with 1000 rows (one per resample) and 10 columns (one per coefficient)
boot_coefs = pd.DataFrame(boot_coefs, columns=topic_cols)

# 95% interval: from the 2.5th to the 97.5th percentile of each coefficient
lower = boot_coefs.quantile(0.025)
upper = boot_coefs.quantile(0.975)

print()
print("Bootstrap 95% intervals (1000 resamples), sorted by coefficient:")
print(f"{'Topic':<18}{'Coef':>7}{'Lower':>8}{'Upper':>8}  Reliability")
for name in coefs.index:  # same order as before: most negative to most positive
    unreliable = lower[name] <= 0 <= upper[name]  # does the interval contain zero?
    flag = "NOT RELIABLE" if unreliable else "reliable"
    print(f"{name:<18}{coefs[name]:>7.2f}{lower[name]:>8.2f}{upper[name]:>8.2f}  {flag}")
