import sys
import json
import pandas as pd
from sklearn.metrics import f1_score
from pipeline import (DATA_FOLDER, MODELS_FOLDER, TOPICS, clean_negative, clean_positive,
                      topics_of_many_texts, split_review, route_pieces, split_full_text)
from report import make_report, make_report_from_full_texts, print_report

# Test of the fixed rule on ONE real hotel, with Booking reviews copied by hand.
# Booking keeps "liked" and "disliked" apart, so we know the right answer: we glue the two parts
# into one text (like a Google review), give it to the fixed rule + Model 2 and compare.
# Unlike verify_full_texts.py, these texts have their REAL punctuation (the 515K has none).
# Input: a CSV with the columns score, liked, disliked, text_if_unknown, fields_known (yes/no).
#   fields_known = no: the copied review had only one block of text and we cannot tell if it was
#   "liked" or "disliked": these reviews are left out of the comparison, but used in the full report.
# Use:  python src/check_booking_hotel.py                       (data/citizenm_zurich_booking.csv)
#       python src/check_booking_hotel.py data/other_hotel.csv

path = DATA_FOLDER.parent / sys.argv[1] if len(sys.argv) > 1 else DATA_FOLDER / "citizenm_zurich_booking.csv"
df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
known = df[df["fields_known"] == "yes"].reset_index(drop=True)
print(f"{path.name}: {len(df)} reviews, {len(known)} with liked / disliked known, mean score {df['score'].mean():.2f}")

liked = list(known["liked"])
disliked = list(known["disliked"])

# ---------- Step 1: side of each piece (the right answer is the Booking block it comes from) ----------
# The two blocks are glued with a new line, which the rule always cuts: splitting each block on its
# own gives exactly the same pieces as splitting the glued text.

rows = []
for i, good, bad in zip(known["review"], liked, disliked):     # i = number of the review in the file
    for block, right in [(good, "positive"), (bad, "negative")]:
        pieces = split_review(block)
        negative, positive, _ = route_pieces(pieces)
        for p in pieces:
            chosen = "negative" if p in negative else ("positive" if p in positive else "aside")
            rows.append({"review": i, "block": "liked" if right == "positive" else "disliked",
                         "piece": p, "right side": right, "chosen": chosen})
pieces = pd.DataFrame(rows)
pieces["ok"] = pieces["right side"] == pieces["chosen"]

print()
print("=== Side of each piece ===")
for block in ["liked", "disliked"]:
    part = pieces[pieces["block"] == block]
    print(f"  from '{block}': {len(part)} pieces, right side {part['ok'].mean() * 100:.1f}% ({part['ok'].sum()})")
print(f"  all: {len(pieces)} pieces, right side {pieces['ok'].mean() * 100:.1f}%")
print()
print("Pieces on the 'wrong' side (check by hand: is the rule wrong, or the guest's block?):")
for _, r in pieces[~pieces["ok"]].iterrows():
    print(f"  review {r['review']:>2} [{r['block']} -> {r['chosen']}] {r['piece']}")
print("(review = number of the review in the file, column 'review')")

# ---------- Step 2: topics and report, compared with the two Booking blocks ----------

def topic_table(neg_texts, pos_texts):
    neg = topics_of_many_texts([clean_negative(t) for t in neg_texts])
    pos = topics_of_many_texts([clean_positive(t) for t in pos_texts])
    return ([[1 if t in x else 0 for t in TOPICS] for x in neg],
            [[1 if t in x else 0 for t in TOPICS] for x in pos])

def flat(table):
    return [v for row in table for v in row]

def no_split(text):
    negative, positive, _ = route_pieces([text])
    return " ".join(negative), " ".join(positive)

def rule(text):
    negative, positive, _, _ = split_full_text(text)
    return negative, positive

weights = pd.read_csv(MODELS_FOLDER / "impact_weights.csv", index_col="topic")["coef"]
with open(MODELS_FOLDER / "impact_intercept.json") as f:
    intercept = json.load(f)["intercept"]

def predicted_score(neg_table, pos_table):
    """intercept + sum of (share of reviews x weight), as in verify_hotels.py."""
    shares = pd.DataFrame([n + p for n, p in zip(neg_table, pos_table)],
                          columns=["Neg_" + t for t in TOPICS] + ["Pos_" + t for t in TOPICS]).mean()
    return intercept + (shares[weights.index] * weights).sum()

ref_neg, ref_pos = topic_table(disliked, liked)
ref_report = make_report(disliked, liked)
results = [{"method": "Reference: Booking blocks", "F1 Neg_ topics": 1.0, "F1 Pos_ topics": 1.0,
            "Fix first": ", ".join(r["name"] for r in ref_report["fix_first"]) or "-",
            "Protect": ", ".join(r["name"] for r in ref_report["protect"]) or "-",
            "predicted score": predicted_score(ref_neg, ref_pos)}]

for glue_name, separator in [("new line", "\n"), ("space", " ")]:
    texts = [separator.join(t for t in [good, bad] if t != "") for good, bad in zip(liked, disliked)]
    for name, method in [("All negative", lambda t: (t, "")), ("Model 2, no split", no_split),
                         ("Fixed rule + Model 2", rule)]:
        out = [method(t) for t in texts]
        neg_texts = [o[0] for o in out]
        pos_texts = [o[1] for o in out]
        neg, pos = topic_table(neg_texts, pos_texts)
        report = make_report(neg_texts, pos_texts)
        results.append({"method": f"{name} ({glue_name})",
                        "F1 Neg_ topics": f1_score(flat(ref_neg), flat(neg)),
                        "F1 Pos_ topics": f1_score(flat(ref_pos), flat(pos)),
                        "Fix first": ", ".join(r["name"] for r in report["fix_first"]) or "-",
                        "Protect": ", ".join(r["name"] for r in report["protect"]) or "-",
                        "predicted score": predicted_score(neg, pos)})

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print()
print(f"=== {len(known)} reviews with liked / disliked known (true mean score {known['score'].mean():.2f}) ===")
print(pd.DataFrame(results).round(3).to_string(index=False))
print("new line = the two blocks on two lines; space = glued with a space (harder: the rule must find the cut)")

# ---------- Step 3: the report the manager would see, ALL reviews as one text each ----------

full_texts = [t if t != "" else "\n".join(x for x in [good, bad] if x != "")
              for good, bad, t in zip(df["liked"], df["disliked"], df["text_if_unknown"])]
print()
print(f"=== Report from ALL {len(df)} reviews, each as one text (what the demo would show) ===")
print_report(make_report_from_full_texts(full_texts))
