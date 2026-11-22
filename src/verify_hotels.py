import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # save the chart to a file, without opening a window
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

# Check of the report on ALL the hotels of the test set (never seen by Model 3):
# predicted score of a hotel = intercept + sum over the 42 topics of (share of reviews x weight)

# Main folder of the repository (one level above the folder of this file):
# the paths work from whatever folder the script is launched
ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT / "data"
MODELS_FOLDER = ROOT / "models"
RESULTS_FOLDER = ROOT / "results"

# ---------- Step 1: load the table, the weights and the intercept ----------

df = pd.read_csv(DATA_FOLDER / "review_topics.csv")       # one row per review, 42 topic columns 0/1
weights = pd.read_csv(MODELS_FOLDER / "impact_weights.csv", index_col="topic")["coef"]
with open(MODELS_FOLDER / "impact_intercept.json") as f:
    intercept = json.load(f)["intercept"]

# ALL the 42 weights, also the not reliable ones: here we check the whole model
topic_cols = list(weights.index)
print("Weights:", len(topic_cols), "- Intercept:", round(intercept, 3))

# ---------- Step 2: keep only the test hotels (same split as impact_full.py) ----------

splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["Hotel_Address"]))
train, test = df.iloc[train_idx], df.iloc[test_idx]

# ---------- Step 3: predicted and true score of each hotel ----------

by_hotel = test.groupby("Hotel_Address")
shares = by_hotel[topic_cols].mean()                      # mean of a 0/1 column = share of the reviews
hotels = pd.DataFrame({
    "n_reviews": by_hotel.size(),
    "true": by_hotel["Reviewer_Score"].mean(),
    "predicted": intercept + (shares * weights).sum(axis=1),
})

# ---------- Step 4: how good is the prediction? ----------

errors = hotels["predicted"] - hotels["true"]
mae = errors.abs().mean()
correlation = hotels["predicted"].corr(hotels["true"])    # Pearson correlation

# Baseline: the same score for every hotel (mean score of the train reviews)
train_mean = train["Reviewer_Score"].mean()
mae_baseline = (train_mean - hotels["true"]).abs().mean()

print("Test hotels:", len(hotels), "- Reviews:", hotels["n_reviews"].sum())
print(f"MAE by hotel: {mae:.3f}  (baseline, always {train_mean:.2f}: {mae_baseline:.3f})")
print(f"Correlation predicted / true: {correlation:.3f}")
print(f"Mean error (predicted - true): {errors.mean():+.3f}")
print(f"Spread (standard deviation): true {hotels['true'].std():.3f}, predicted {hotels['predicted'].std():.3f}")

# ---------- Step 5: scatter plot ----------

BROWN, CREAM, TEXT = "#7a5c46", "#f6f2ec", "#2b2420"      # colors of Roomor

# The same limits on the two axes, so the diagonal is really at 45 degrees
# (a small margin of 0.1, then rounded to the nearest half point outwards)
low = np.floor((hotels[["true", "predicted"]].min().min() - 0.1) * 2) / 2
high = np.ceil((hotels[["true", "predicted"]].max().max() + 0.1) * 2) / 2

fig, ax = plt.subplots(figsize=(7, 7), facecolor=CREAM)
ax.set_facecolor(CREAM)
ax.plot([low, high], [low, high], color=TEXT, linewidth=1.5, linestyle="--", alpha=0.6, zorder=1)
ax.scatter(hotels["predicted"], hotels["true"], s=55, color=BROWN, alpha=0.75,
           edgecolor=CREAM, linewidth=1, zorder=2)
ax.text(low + 0.25, low + 0.36, "predicted = true", color=TEXT, fontsize=10,
        rotation=45, rotation_mode="anchor")

ax.set_xlim(low, high)
ax.set_ylim(low, high)
ax.set_aspect("equal")
ax.grid(color=TEXT, alpha=0.1, linewidth=0.8)
ax.set_axisbelow(True)
for side in ["top", "right", "left", "bottom"]:
    ax.spines[side].set_visible(False)
ax.tick_params(colors=TEXT, length=0, pad=8)
ax.set_xlabel("Predicted mean score (from the topics in the reviews)", color=TEXT, fontsize=11)
ax.set_ylabel("True mean score", color=TEXT, fontsize=11)
ax.text(0, 1.065, "Predicted and true score of hotels never seen by the model", color=TEXT,
        fontsize=13, fontweight="bold", transform=ax.transAxes)
ax.text(0, 1.025, f"{len(hotels)} test hotels, one dot each · MAE {mae:.2f} · correlation {correlation:.2f}",
        color=TEXT, fontsize=10, transform=ax.transAxes)
fig.tight_layout()

RESULTS_FOLDER.mkdir(exist_ok=True)
fig.savefig(RESULTS_FOLDER / "hotel_scores.png", dpi=150, facecolor=CREAM)
print("Saved results/hotel_scores.png")
