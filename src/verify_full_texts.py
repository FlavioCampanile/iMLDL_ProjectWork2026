import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import f1_score
from pipeline import (DATA_FOLDER, MODELS_FOLDER, TOPICS, clean_negative, clean_positive,
                      topics_of_many_texts, split_review, route_pieces, split_full_text)
from report import make_report

# Check of the fixed rule (reviews written as ONE text, like on Google or TripAdvisor).
# Idea: Booking reviews already have the right answer. We glue "liked" and "disliked" into one text,
# give it to the fixed rule + Model 2, and compare what comes out with the two original fields.
# Methods compared (each one rebuilds a negative and a positive text from the glued review):
#   - "All negative" (baseline): the whole text goes to the negative side (what happens today)
#   - "Model 2, no split" (baseline): the whole text is ONE piece, Model 2 decides its side
#   - "Fixed rule + Model 2": split_full_text in pipeline.py (the new feature)
# Reference = the two Booking fields: it is the best possible result, no method can beat it.

RESULTS_FOLDER = DATA_FOLDER.parent / "results"
rng = np.random.RandomState(42)

# ---------- Step 1: the 224 test hotels (same split as impact_full.py and report.py) ----------

df = pd.read_csv(DATA_FOLDER / "Hotel_Reviews.csv")
df = df.drop_duplicates()
for col in ["Hotel_Address", "Negative_Review", "Positive_Review"]:
    df[col] = df[col].str.strip()
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["Hotel_Address"]))
test = df.iloc[test_idx].reset_index(drop=True)
print("Test hotels:", test["Hotel_Address"].nunique(), "- reviews:", len(test))

# The two fields cleaned as in apply_topics.py ("No Negative" -> ""): this is the reference
ref_neg = [clean_negative(t) for t in test["Negative_Review"]]
ref_pos = [clean_positive(t) for t in test["Positive_Review"]]

# ---------- Step 2: glue the two fields into one text ----------
# The order (liked first or disliked first) is chosen at random for each review.
# Two versions, because the 515K has no punctuation inside the fields:
#   A: a full stop between the two fields (like a real review, where a new sentence starts)
#   B: no punctuation at all (the hardest case: only the contrast words can cut the text)

def glue(neg, pos, separator, negative_first):
    """Return the glued text and, for each word, the field it comes from ("neg" or "pos")."""
    parts = [(neg, "neg"), (pos, "pos")]
    if not negative_first:
        parts = parts[::-1]
    parts = [(text, side) for text, side in parts if text != ""]
    text = separator.join(t for t, _ in parts)
    labels = [side for t, side in parts for _ in t.split()]
    return text, labels

negative_first = rng.rand(len(test)) < 0.5
glued = {}
for version, separator in [("A (full stop)", ". "), ("B (no punctuation)", " ")]:
    glued[version] = [glue(n, p, separator, f) for n, p, f in zip(ref_neg, ref_pos, negative_first)]

# ---------- Step 3: the three methods ----------

def method_all_negative(text):
    return text, ""

def method_no_split(text):
    """The whole review is one piece: Model 2 sends it to one side (or puts it aside)."""
    negative, positive, _ = route_pieces([text]) if text != "" else ([], [], [])
    return " ".join(negative), " ".join(positive)

def method_rule(text):
    negative, positive, _, _ = split_full_text(text)
    return negative, positive

METHODS = {"All negative": method_all_negative,
           "Model 2, no split": method_no_split,
           "Fixed rule + Model 2": method_rule}

# ---------- Step 4: side of each piece (only for the fixed rule) ----------

def piece_sides(text, labels):
    """Split the glued text with the fixed rule, send the pieces to a side with Model 2 and
    return, for each piece: (true field, side chosen). The true field is the field of most of
    its words; "mixed" if the piece has words of both fields."""
    pieces = split_review(text)
    negative, positive, aside = route_pieces(pieces)
    chosen = {}
    for p in negative:
        chosen.setdefault(p, []).append("neg")
    for p in positive:
        chosen.setdefault(p, []).append("pos")
    for p in aside:
        chosen.setdefault(p, []).append("aside")
    result = []
    position = 0                                  # the words of the pieces follow the glued text in order
    for p in pieces:
        n = len(p.split())
        words_labels = labels[position:position + n]
        position += n
        true = words_labels[0] if len(set(words_labels)) == 1 else "mixed"
        result.append((true, chosen[p].pop(0)))
    return result

# ---------- Step 5: run everything and compute the metrics ----------

weights = pd.read_csv(MODELS_FOLDER / "impact_weights.csv", index_col="topic")["coef"]
with open(MODELS_FOLDER / "impact_intercept.json") as f:
    intercept = json.load(f)["intercept"]

def topic_matrix(neg_texts, pos_texts):
    """0/1 table with the 21 Neg_ and the 21 Pos_ columns (same as apply_topics.py)."""
    neg_topics = topics_of_many_texts([clean_negative(t) for t in neg_texts])
    pos_topics = topics_of_many_texts([clean_positive(t) for t in pos_texts])
    table = pd.DataFrame({"Neg_" + t: [1 if t in x else 0 for x in neg_topics] for t in TOPICS})
    for t in TOPICS:
        table["Pos_" + t] = [1 if t in x else 0 for x in pos_topics]
    return table

def hotel_scores(table):
    """Predicted mean score of each hotel = intercept + sum of (share x weight), as in verify_hotels.py."""
    shares = table.groupby(test["Hotel_Address"].values).mean()
    return intercept + (shares[weights.index] * weights).sum(axis=1)

def report_topics(neg_texts, pos_texts):
    """For each hotel: the list of topics in Fix first and in Protect (report.py)."""
    result = {}
    for address, rows in test.groupby("Hotel_Address").groups.items():
        report = make_report([neg_texts[i] for i in rows], [pos_texts[i] for i in rows])
        result[address] = ([r["topic"] for r in report["fix_first"]], [r["topic"] for r in report["protect"]])
    return result

def jaccard(a, b):
    """Share of topics in common between two lists (1 if both are empty)."""
    a, b = set(a), set(b)
    return 1.0 if len(a | b) == 0 else len(a & b) / len(a | b)

true_scores = test.groupby("Hotel_Address")["Reviewer_Score"].mean()
neg_cols = ["Neg_" + t for t in TOPICS]
pos_cols = ["Pos_" + t for t in TOPICS]

print("Reference (the two Booking fields)...")
ref_table = topic_matrix(ref_neg, ref_pos)
ref_scores = hotel_scores(ref_table)
ref_reports = report_topics(ref_neg, ref_pos)

rows = [{"version": "-", "method": "Reference: Booking fields",
         "F1 Neg_ topics": 1.0, "F1 Pos_ topics": 1.0,
         "MAE by hotel": (ref_scores - true_scores).abs().mean(),
         "correlation": ref_scores.corr(true_scores),
         "same #1 in Fix first": 1.0, "Fix first overlap": 1.0, "Protect overlap": 1.0}]

for version, items in glued.items():
    texts = [t for t, _ in items]
    for name, method in METHODS.items():
        print(f"{version} - {name}...")
        out = [method(t) for t in texts]
        neg_texts = [n for n, _ in out]
        pos_texts = [p for _, p in out]
        table = topic_matrix(neg_texts, pos_texts)
        scores = hotel_scores(table)
        reports = report_topics(neg_texts, pos_texts)
        same_first, fix_overlap, protect_overlap = [], [], []
        for address, (ref_fix, ref_protect) in ref_reports.items():
            fix, protect = reports[address]
            same_first.append((ref_fix[:1] == fix[:1]))
            fix_overlap.append(jaccard(ref_fix, fix))
            protect_overlap.append(jaccard(ref_protect, protect))
        rows.append({"version": version, "method": name,
                     "F1 Neg_ topics": f1_score(ref_table[neg_cols].values.ravel(), table[neg_cols].values.ravel()),
                     "F1 Pos_ topics": f1_score(ref_table[pos_cols].values.ravel(), table[pos_cols].values.ravel()),
                     "MAE by hotel": (scores - true_scores).abs().mean(),
                     "correlation": scores.corr(true_scores),
                     "same #1 in Fix first": np.mean(same_first),
                     "Fix first overlap": np.mean(fix_overlap),
                     "Protect overlap": np.mean(protect_overlap)})

summary = pd.DataFrame(rows)
baseline_mae = (test["Reviewer_Score"].mean() - true_scores).abs().mean()   # only for orientation
pd.set_option("display.width", 200)
print()
print("=== Comparison with the two Booking fields, 224 test hotels ===")
print(summary.round(3).to_string(index=False))
print("(MAE by hotel of the constant baseline 8.37 in verify_hotels.py: 0.523)")

# Side of each piece, fixed rule only
print()
print("=== Side chosen for each piece by the fixed rule + Model 2 ===")
for version, items in glued.items():
    sides = [s for text, labels in items for s in piece_sides(text, labels)]
    sides = pd.DataFrame(sides, columns=["true", "chosen"])
    print(version, "- pieces:", len(sides), f"- mixed (words of both fields): {(sides['true'] == 'mixed').mean() * 100:.1f}%")
    for true in ["neg", "pos"]:
        part = sides[sides["true"] == true]["chosen"].value_counts(normalize=True) * 100
        print(f"  from the {true} field ({(sides['true'] == true).sum()} pieces): "
              f"right side {part.get(true, 0):.1f}%, wrong side {part.get('pos' if true == 'neg' else 'neg', 0):.1f}%, "
              f"put aside {part.get('aside', 0):.1f}%")

# ---------- Step 6: HRAST, sentences WITH punctuation (closer to a real Google review) ----------
# The test sentences of Model 2 (same filters and split as sentiment_model.py, never seen in training)
# are glued in groups of 4 into invented "reviews"; each sentence keeps its true sentiment.

hrast = pd.read_csv(DATA_FOLDER / "HRAST.csv", encoding="utf-8-sig").drop(columns=["Unnamed: 2"])
mixed = hrast[(hrast["Aspect"] == 1)]["review"].drop_duplicates().tolist()   # never used to train Model 2
simple = hrast.drop_duplicates(subset="review")
simple = simple[(simple["positive"] + simple["negative"] + simple["neutral"] == 1) & (simple["neutral"] == 0)
                & (simple["Aspect"] == 0)]
_, X_test, _, y_test = train_test_split(simple["review"], simple["positive"], test_size=0.15,
                                        random_state=42, stratify=simple["positive"])
sentences = list(X_test)
labels = ["pos" if y == 1 else "neg" for y in y_test]
order = rng.permutation(len(sentences))
results = {"All negative": [], "Model 2, no split": [], "Fixed rule + Model 2": []}
for start in range(0, len(order), 4):
    group = order[start:start + 4]
    text_parts = []
    for i in group:
        s = sentences[i].strip()
        if s[-1] not in ".!?":
            s += "."                               # every sentence ends with a full stop
        text_parts.append(s)
    text = " ".join(text_parts)
    group_labels = [labels[i] for i in group]
    # All negative: every sentence on the negative side
    results["All negative"] += [(t, "neg") for t in group_labels]
    # No split: the whole text gets the side of Model 2
    negative, positive, aside = route_pieces([text])
    side = "neg" if negative else ("pos" if positive else "aside")
    results["Model 2, no split"] += [(t, side) for t in group_labels]
    # Fixed rule: each sentence is cut on its own (the full stops are always cut), so the pieces
    # of one sentence keep its true sentiment
    for s, i in zip(text_parts, group):
        pieces = split_review(s)
        neg_p, pos_p, aside_p = route_pieces(pieces)
        for p in pieces:
            chosen = "neg" if p in neg_p else ("pos" if p in pos_p else "aside")
            results["Fixed rule + Model 2"].append((labels[i], chosen))

print()
print(f"=== HRAST: {len(sentences)} test sentences of Model 2, glued 4 at a time ===")
for name, pairs in results.items():
    pairs = pd.DataFrame(pairs, columns=["true", "chosen"])
    right = (pairs["true"] == pairs["chosen"]).mean() * 100
    aside = (pairs["chosen"] == "aside").mean() * 100
    neg = pairs[pairs["true"] == "neg"]
    neg_right = (neg["chosen"] == "neg").mean() * 100
    print(f"  {name:<22} right side {right:5.1f}% - put aside {aside:4.1f}% - criticisms on the negative side {neg_right:5.1f}%")

# Mixed sentences (Aspect = 1: praise AND criticism in the same sentence, e.g. "nice room but noisy")
split_count, both_sides = 0, 0
for s in mixed:
    pieces = split_review(s)
    neg_p, pos_p, _ = route_pieces(pieces)
    split_count += len(pieces) >= 2
    both_sides += len(neg_p) > 0 and len(pos_p) > 0
print()
print(f"=== HRAST: {len(mixed)} mixed sentences (Aspect = 1) ===")
print(f"  cut into 2 or more pieces by the rule: {split_count / len(mixed) * 100:.1f}%")
print(f"  with pieces on BOTH sides (praise and criticism separated): {both_sides / len(mixed) * 100:.1f}%"
      " (without the rule: 0%, a sentence has one side only)")

RESULTS_FOLDER.mkdir(exist_ok=True)
summary.to_csv(RESULTS_FOLDER / "full_texts_check.csv", index=False)
print()
print("Saved results/full_texts_check.csv")
