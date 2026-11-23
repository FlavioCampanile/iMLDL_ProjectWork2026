import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # save the chart to a file, without opening a window
import matplotlib.pyplot as plt
from pipeline import MODELS_FOLDER, DATA_FOLDER, TOPICS, clean_negative, clean_positive, \
    topics_of_many_texts, split_full_text
from report import READABLE_NAMES, load_weights

# What Model 3 says about ONE hotel, review by review and topic by topic.
# Each review (one whole text, like on Google) goes through the fixed rule + Model 2 (negative /
# positive side) and Model 1 (topics); Model 3 turns the topics into points of score.
# Input: an Excel file with a column "text" (one review per row) and a column "score" (the guest's score).
# Use:  python src/explain_hotel.py                          (data/google_check.xlsx)
#       python src/explain_hotel.py data/other.xlsx "Hotel name"

path = DATA_FOLDER.parent / sys.argv[1] if len(sys.argv) > 1 else DATA_FOLDER / "google_check.xlsx"
hotel_name = sys.argv[2] if len(sys.argv) > 2 else "citizenM Zürich"
table = pd.read_excel(path)
texts = [str(t) for t in table["text"]]
scores = table["score"].astype(float).to_numpy()

# ---------- Step 1: Model 2 + fixed rule, then Model 1 ----------

sides = [split_full_text(t) for t in texts]                 # (negative text, positive text, pieces, aside)
neg_topics = topics_of_many_texts([clean_negative(s[0]) for s in sides])
pos_topics = topics_of_many_texts([clean_positive(s[1]) for s in sides])

# ---------- Step 2: Model 3, review by review ----------
# predicted score = intercept + weights of the topics criticized + weights of the topics praised
# (ALL 42 weights, as in verify_hotels.py: here we look at the whole model)

weights = pd.read_csv(MODELS_FOLDER / "impact_weights.csv", index_col="topic")["coef"]
with open(MODELS_FOLDER / "impact_intercept.json") as f:
    intercept = json.load(f)["intercept"]
TRAIN_MEAN = 8.372                    # baseline of impact_full.py: mean score of the train reviews

rows = []
for i, (neg, pos) in enumerate(zip(neg_topics, pos_topics)):
    predicted = intercept + sum(weights["Neg_" + t] for t in neg) + sum(weights["Pos_" + t] for t in pos)
    rows.append({"review": i + 1, "score": scores[i], "predicted": predicted,
                 "error": predicted - scores[i],
                 "criticized": ", ".join(f"{READABLE_NAMES[t]} {weights['Neg_' + t]:+.2f}" for t in neg) or "-",
                 "praised": ", ".join(f"{READABLE_NAMES[t]} {weights['Pos_' + t]:+.2f}" for t in pos) or "-"})
reviews = pd.DataFrame(rows)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 90)
print(f"{hotel_name}: {len(reviews)} reviews - intercept {intercept:.2f}")
print()
print("=== Review by review: score of the guest, score predicted by Model 3 and why ===")
print(reviews.round(2).to_string(index=False))

# ---------- Step 3: the whole hotel ----------

mae_model = reviews["error"].abs().mean()
mae_base = np.abs(TRAIN_MEAN - scores).mean()
print()
print("=== The hotel ===")
print(f"True mean score:        {scores.mean():.2f}")
print(f"Predicted mean score:   {reviews['predicted'].mean():.2f}")
print(f"MAE per review: Model 3 {mae_model:.2f} - baseline (always {TRAIN_MEAN:.2f}) {mae_base:.2f}")
print(f"Correlation predicted / true, per review: {reviews['predicted'].corr(reviews['score']):.2f}")

# ---------- Step 4: points by topic (what the report ranks) ----------
# points = share of reviews x weight. Only the reliable weights (report.py), so the chart and the
# report agree.

reliable, averages = load_weights()
topic_rows = []
for t in TOPICS:
    for side, found in [("Neg_", neg_topics), ("Pos_", pos_topics)]:
        col = side + t
        share = sum(1 for x in found if t in x) / len(found)
        if share == 0 or col not in reliable:
            continue
        topic_rows.append({"topic": READABLE_NAMES[t], "side": "criticized" if side == "Neg_" else "praised",
                           "share": share, "average share": averages[col] / 100, "weight": reliable[col],
                           "points": share * reliable[col],
                           "vs average hotel": (share - averages[col] / 100) * reliable[col]})
topics = pd.DataFrame(topic_rows).sort_values("points")
print()
print("=== Points of score by topic (share of reviews x weight), reliable weights only ===")
print(topics.round(3).to_string(index=False))
lost = topics.loc[topics["points"] < 0, "points"].sum()
gained = topics.loc[topics["points"] > 0, "points"].sum()
print(f"Total: {lost:+.2f} lost to criticism, {gained:+.2f} gained from praise")

# ---------- Step 5: chart ----------
# One bar per topic and side: left = points lost to criticism, right = points gained from praise.

CREAM, TEXT, MUTED = "#f6f2ec", "#2b2420", "#6f655d"
LOST, GAINED = "#a3502f", "#007a99"           # diverging pair: warm = lost, cool = gained

# Only the topics worth at least 0.01 points (the others would be invisible bars)
topics = topics[topics["points"].abs() >= 0.01].copy()
topics["label"] = topics["topic"] + np.where(topics["side"] == "criticized", " (criticized)", " (praised)")
topics = topics.sort_values("points", ascending=False)
fig, ax = plt.subplots(figsize=(8, 0.42 * len(topics) + 1.6), facecolor=CREAM)
ax.set_facecolor(CREAM)
y = np.arange(len(topics))
colors = [LOST if p < 0 else GAINED for p in topics["points"]]
ax.barh(y, topics["points"], color=colors, height=0.62, edgecolor=CREAM, linewidth=2)
for yi, p in zip(y, topics["points"]):
    ax.text(p + (0.006 if p >= 0 else -0.006), yi, f"{p:+.2f}", va="center",
            ha="left" if p >= 0 else "right", color=TEXT, fontsize=9)
ax.set_yticks(y, topics["label"], color=TEXT, fontsize=9.5)
ax.invert_yaxis()
ax.axvline(0, color=TEXT, linewidth=1)
limit = topics["points"].abs().max() * 1.35
ax.set_xlim(-limit, limit)
ax.grid(axis="x", color=TEXT, alpha=0.08)
ax.set_axisbelow(True)
for s in ["top", "right", "left", "bottom"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", colors=MUTED, length=0)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Points of score (share of reviews x weight of Model 3)", color=MUTED, fontsize=9.5)
fig.text(0.02, 0.985, f"{hotel_name}: where the score is lost and gained", color=TEXT,
         fontsize=13, fontweight="bold", va="top")
fig.text(0.02, 0.945, f"{len(reviews)} reviews · true mean {scores.mean():.2f} · predicted {reviews['predicted'].mean():.2f}"
         " · topics worth at least 0.01 points · relative priorities, not points to win back", color=MUTED, fontsize=9, va="top")
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = DATA_FOLDER.parent / "results" / (hotel_name.lower().replace(" ", "_").replace("ü", "u") + "_points.png")
fig.savefig(out, dpi=150, facecolor=CREAM)
print()
print("Saved", out)
