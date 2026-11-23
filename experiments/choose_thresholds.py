import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

# The functions of the pipeline are in src/: add that folder to the places where Python looks
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import DATA_FOLDER, TOPICS, clean_negative, clean_positive, topics_of_many_texts, split_full_text
from report import make_report

# Choice of the settings of the fixed rule ON THE TRAIN HOTELS (validation), never on the test hotels.
# Settings tried: the two thresholds of Model 2 (below LOW -> negative, above HIGH -> positive,
# in between -> put aside) and whether Model 2 reads the piece without the contrast word at its start.
# Rule decided BEFORE looking at the results:
#   - criterion: mean overlap between the "Fix first" topics of the glued review and of the two Booking
#     fields (same measure as verify_full_texts.py, version A = full stop between the two fields);
#   - a setting replaces the current one (0.35 / 0.65, contrast word kept) only if it is better by
#     more than 0.01; otherwise we keep the current one (simpler, decided before).

SETTINGS = [(low, high, strip) for strip in [False, True]
            for low, high in [(0.30, 0.70), (0.35, 0.65), (0.40, 0.60), (0.45, 0.55), (0.50, 0.50)]]
CURRENT = (0.35, 0.65, False)
N_HOTELS = 150                      # train hotels drawn at random (all of them would take too long)

# ---------- Step 1: 150 train hotels (same split as impact_full.py: these are NOT the test hotels) ----------

df = pd.read_csv(DATA_FOLDER / "Hotel_Reviews.csv")
df = df.drop_duplicates()
for col in ["Hotel_Address", "Negative_Review", "Positive_Review"]:
    df[col] = df[col].str.strip()
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["Hotel_Address"]))
train = df.iloc[train_idx]

rng = np.random.RandomState(42)
hotels = rng.choice(sorted(train["Hotel_Address"].unique()), size=N_HOTELS, replace=False)
val = train[train["Hotel_Address"].isin(hotels)].reset_index(drop=True)
print("Validation: train hotels", val["Hotel_Address"].nunique(), "- reviews", len(val))

# ---------- Step 2: reference (Booking fields) and glued texts (version A) ----------

ref_neg = [clean_negative(t) for t in val["Negative_Review"]]
ref_pos = [clean_positive(t) for t in val["Positive_Review"]]
negative_first = rng.rand(len(val)) < 0.5
glued = []
for n, p, first in zip(ref_neg, ref_pos, negative_first):
    parts = [n, p] if first else [p, n]
    glued.append(". ".join(t for t in parts if t != ""))

def fix_first_topics(neg_texts, pos_texts):
    """For each hotel: the topics of Fix first (report.py)."""
    return {address: [r["topic"] for r in make_report([neg_texts[i] for i in rows],
                                                      [pos_texts[i] for i in rows])["fix_first"]]
            for address, rows in val.groupby("Hotel_Address").groups.items()}

def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if len(a | b) == 0 else len(a & b) / len(a | b)

ref_fix = fix_first_topics(ref_neg, ref_pos)
ref_neg_topics = topics_of_many_texts(ref_neg)

# ---------- Step 3: try every setting ----------

rows = []
for low, high, strip in SETTINGS:
    out = [split_full_text(t, low, high, strip) for t in glued]
    neg_texts = [o[0] for o in out]
    pos_texts = [o[1] for o in out]
    fix = fix_first_topics(neg_texts, pos_texts)
    neg_topics = topics_of_many_texts([clean_negative(t) for t in neg_texts])
    y_ref = [[1 if t in x else 0 for t in TOPICS] for x in ref_neg_topics]
    y_new = [[1 if t in x else 0 for t in TOPICS] for x in neg_topics]
    rows.append({"low": low, "high": high, "strip contrast word": strip,
                 "Fix first overlap": np.mean([jaccard(ref_fix[a], fix[a]) for a in ref_fix]),
                 "same #1 in Fix first": np.mean([ref_fix[a][:1] == fix[a][:1] for a in ref_fix]),
                 "F1 Neg_ topics": f1_score(np.ravel(y_ref), np.ravel(y_new)),
                 "% pieces put aside": sum(o[3] for o in out) / sum(o[2] for o in out) * 100})
    print(f"  {low:.2f} / {high:.2f}, strip={strip}: overlap {rows[-1]['Fix first overlap']:.3f}")

results = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print()
print(results.round(3).to_string(index=False))

# ---------- Step 4: apply the rule decided before ----------

current = results[(results["low"] == CURRENT[0]) & (results["high"] == CURRENT[1])
                  & (results["strip contrast word"] == CURRENT[2])].iloc[0]
best = results.loc[results["Fix first overlap"].idxmax()]
print()
print(f"Current setting: overlap {current['Fix first overlap']:.3f}")
print(f"Best setting: {best['low']:.2f} / {best['high']:.2f}, strip={best['strip contrast word']}, "
      f"overlap {best['Fix first overlap']:.3f}")
if best["Fix first overlap"] - current["Fix first overlap"] > 0.01:
    print("-> The best setting wins by more than 0.01: put it in pipeline.py")
else:
    print("-> Difference of 0.01 or less: keep the current setting")
