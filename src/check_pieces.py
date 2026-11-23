import sys
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pipeline import (DATA_FOLDER, CONTRAST_WORDS, sentiment_model, sentiment_vectorizer,
                      clean_negative, clean_positive, split_review, route_pieces, text_for_sentiment,
                      STRIP_CONTRAST_WORD)
from report import make_report_from_full_texts, print_report

# Prepares the MANUAL check of the fixed rule: one Excel file, data/manual_check.xlsx, with 3 sheets.
#   google_pieces       every piece of the reviews you copied by hand (data/google_check.xlsx or .csv)
#   wrong_side          30 pieces of the 515K that ended on the "wrong" side (train hotels only)
#   contrast_sentences  30 HRAST sentences with "but", "however"... and how the rule cut them
# The columns in CAPITALS are empty: you fill them in Excel.
# Use:  python src/check_pieces.py                  (reads data/google_check.xlsx or data/google_check.csv)
#       python src/check_pieces.py data/other.xlsx  (another file, e.g. with German reviews)

rng = np.random.RandomState(42)

def p_positive(pieces):
    """Model 2 probability of "positive" for each piece, read exactly as route_pieces reads it."""
    texts = [text_for_sentiment(p) for p in pieces] if STRIP_CONTRAST_WORD else pieces
    return sentiment_model.predict_proba(sentiment_vectorizer.transform(texts))[:, 1]

def side_of(piece, negative, positive):
    return "negative" if piece in negative else ("positive" if piece in positive else "aside")

# ---------- Sheet 1: the reviews copied by hand ----------

def read_reviews(path):
    """Read the reviews: one per row, in a column called "text" (Excel file or CSV)."""
    if path.suffix == ".xlsx":
        table = pd.read_excel(path)
    else:
        # Excel in Italian saves CSV files with ";", in English with ",": try both
        for separator in [";", ","]:
            try:
                table = pd.read_csv(path, sep=separator, encoding="utf-8-sig")
            except pd.errors.ParserError:
                continue
            if "text" in table.columns:
                break
    if "text" not in table.columns:
        raise ValueError(f"{path.name} needs a column called 'text' (one review per row)")
    return [t for t in table["text"].tolist() if isinstance(t, str) and t.strip() != ""]

if len(sys.argv) > 1:
    google_path = DATA_FOLDER.parent / sys.argv[1]
else:
    google_path = DATA_FOLDER / "google_check.xlsx"
    if not google_path.exists():
        google_path = DATA_FOLDER / "google_check.csv"

sheets = {}
if google_path.exists():
    reviews = read_reviews(google_path)
    rows = []
    for i, text in enumerate(reviews, start=1):
        pieces = split_review(text)
        negative, positive, _ = route_pieces(pieces)
        for piece, p in zip(pieces, p_positive(pieces) if pieces else []):
            rows.append({"review": i, "piece": piece, "side chosen": side_of(piece, negative, positive),
                         "P(positive)": round(float(p), 3),
                         "CUT OK? (yes/no)": "", "TRUE SIDE (negative/positive/neutral)": "", "NOTES": ""})
    sheets["google_pieces"] = pd.DataFrame(rows)
    sheets["google_reviews"] = pd.DataFrame({"review": range(1, len(reviews) + 1), "text": reviews})
    print(f"{google_path.name}: {len(reviews)} reviews, {len(rows)} pieces")
    print()
    print_report(make_report_from_full_texts(reviews))
else:
    print(f"No file with your reviews ({google_path.name}): sheet google_pieces skipped")

# ---------- Sheet 2: pieces on the "wrong" side, 515K train hotels ----------
# Same glue as verify_full_texts.py, version A (a full stop between the two fields): the full stop
# is always cut, so splitting each field on its own gives exactly the same pieces.
# Question for you: is the piece REALLY on the wrong side, or did the guest write praise
# in the "disliked" field (then the Booking field is wrong, not the rule)?

df = pd.read_csv(DATA_FOLDER / "Hotel_Reviews.csv").drop_duplicates()
for col in ["Hotel_Address", "Negative_Review", "Positive_Review"]:
    df[col] = df[col].str.strip()
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, _ = next(splitter.split(df, groups=df["Hotel_Address"]))
sample = df.iloc[train_idx].sample(3000, random_state=42)

wrong = []
for field, clean, right in [("Negative_Review", clean_negative, "negative"),
                            ("Positive_Review", clean_positive, "positive")]:
    for text in sample[field]:
        pieces = split_review(clean(text))
        negative, positive, _ = route_pieces(pieces)
        for piece in pieces:
            chosen = side_of(piece, negative, positive)
            if chosen != right and len(piece.split()) >= 2:
                wrong.append({"Booking field": "disliked" if right == "negative" else "liked",
                              "piece": piece, "side chosen": chosen})
wrong = pd.DataFrame(wrong)
# 15 from each field (or fewer if there are not enough)
parts = [g.sample(min(15, len(g)), random_state=42) for _, g in wrong.groupby("Booking field")]
wrong = pd.concat(parts).reset_index(drop=True)
wrong["P(positive)"] = np.round(p_positive(wrong["piece"].tolist()), 3)
wrong["TRUE SIDE (negative/positive/neutral)"] = ""
wrong["NOTES"] = ""
sheets["wrong_side"] = wrong
print()
print(f"wrong_side: {len(wrong)} pieces (from 3,000 reviews of train hotels)")

# ---------- Sheet 3: HRAST sentences with a contrast word ----------
# Aspect = 1 (more than one topic) AND a contrast word: the sentences most likely to mix
# praise and criticism. Question: did the rule cut them in the right place and send each part
# to the right side?

hrast = pd.read_csv(DATA_FOLDER / "HRAST.csv", encoding="utf-8-sig")
pattern = r"\b(?:" + "|".join(CONTRAST_WORDS) + r")\b"
candidates = hrast[(hrast["Aspect"] == 1)]["review"].drop_duplicates()
candidates = candidates[candidates.str.contains(pattern, case=False, regex=True)]
rows = []
for sentence in candidates.sample(30, random_state=42):
    pieces = split_review(sentence)
    negative, positive, _ = route_pieces(pieces)
    rows.append({"sentence": sentence,
                 "pieces and sides": "  |  ".join(f"[{side_of(p, negative, positive)}] {p}" for p in pieces),
                 "ALL RIGHT? (yes/no)": "", "WHAT IS WRONG": ""})
sheets["contrast_sentences"] = pd.DataFrame(rows)
print(f"contrast_sentences: {len(rows)} sentences")

# ---------- Save ----------

output = DATA_FOLDER / "manual_check.xlsx"
with pd.ExcelWriter(output) as writer:
    for name, table in sheets.items():
        table.to_excel(writer, sheet_name=name, index=False)
print()
print("Saved", output, "- sheets:", ", ".join(sheets))
