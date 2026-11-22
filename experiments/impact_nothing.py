import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Step 6: how much do the Neg_ coefficients change with the "nothing" rule?
# Same regression as Step 3 of impact_full.py, on two tables built by apply_topics.py

df_off = pd.read_csv("Data/review_topics.csv")            # rule switched off (original table)
df_on = pd.read_csv("Data/review_topics_nothing.csv")     # rule switched on

neg_cols = [col for col in df_off.columns if col.startswith("Neg_")]
pos_cols = [col for col in df_off.columns if col.startswith("Pos_")]
topic_cols = neg_cols + pos_cols

# ---------- Checks: the two tables must differ ONLY because of the rule ----------

same_rows = (df_off["Hotel_Address"].equals(df_on["Hotel_Address"])
             and df_off["Reviewer_Score"].equals(df_on["Reviewer_Score"]))
assert same_rows, "The two tables do not have the same reviews in the same order"
assert df_off[pos_cols].equals(df_on[pos_cols]), "The Pos_ columns should be identical"

# The rule makes a negative text empty: a changed review must have NO Neg_ topic left
changed = (df_off[neg_cols] != df_on[neg_cols]).any(axis=1)
assert (df_on.loc[changed, neg_cols].sum(axis=1) == 0).all(), "A changed review still has Neg_ topics"

print("Reviews:", len(df_off))
print("Reviews changed by the rule:", changed.sum(), f"({changed.mean() * 100:.2f}%)")
print("Mean score of the changed reviews:", round(df_off.loc[changed, "Reviewer_Score"].mean(), 2),
      "- of all reviews:", round(df_off["Reviewer_Score"].mean(), 2))

# ---------- Same split by hotel as in impact_full.py ----------

splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(splitter.split(df_off[topic_cols], df_off["Reviewer_Score"],
                                          df_off["Hotel_Address"]))
print("Train reviews:", len(train_idx), "- Test reviews:", len(test_idx))

# ---------- Regression on the two tables ----------

def fit_regression(df):
    """Train on the train hotels; return the coefficients, the intercept, and MAE and R2 on the test hotels."""
    X, y = df[topic_cols], df["Reviewer_Score"]
    model = LinearRegression()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = model.predict(X.iloc[test_idx])
    coefs = pd.Series(model.coef_, index=topic_cols)
    return (coefs, model.intercept_,
            mean_absolute_error(y.iloc[test_idx], y_pred), r2_score(y.iloc[test_idx], y_pred))

coefs_off, intercept_off, mae_off, r2_off = fit_regression(df_off)
coefs_on, intercept_on, mae_on, r2_on = fit_regression(df_on)

print()
print(f"{'Version':<14}{'Intercept':>11}{'MAE':>9}{'R2':>9}")
print(f"{'Rule off':<14}{intercept_off:>11.3f}{mae_off:>9.4f}{r2_off:>9.4f}")
print(f"{'Rule on':<14}{intercept_on:>11.3f}{mae_on:>9.4f}{r2_on:>9.4f}")

# ---------- Table of the Neg_ coefficients, side by side ----------

diff = coefs_on - coefs_off
share_off = df_off[neg_cols].iloc[train_idx].mean() * 100   # % of train reviews where the column is 1
share_on = df_on[neg_cols].iloc[train_idx].mean() * 100

print()
print(f"{'Topic':<32}{'Off':>8}{'On':>8}{'On - Off':>10}{'% off':>8}{'% on':>8}")
for col in coefs_off[neg_cols].sort_values().index:
    print(f"{col:<32}{coefs_off[col]:>8.3f}{coefs_on[col]:>8.3f}{diff[col]:>10.3f}"
          f"{share_off[col]:>8.2f}{share_on[col]:>8.2f}")

biggest = diff[neg_cols].abs().idxmax()
print()
print(f"Biggest difference: {biggest} ({coefs_off[biggest]:.3f} -> {coefs_on[biggest]:.3f}, "
      f"{diff[biggest]:+.3f})")
