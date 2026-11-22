from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Saves the train / validation / test sentences of Model 1 in three CSV files,
# so DistilBERT can be trained and tested on EXACTLY the same sentences.
# Step 1 and Step 2 are the same as in topic_model.py: if one changes, change the other too.

# Main folder of the repository (one level above the folder of this file):
# the paths work from whatever folder the script is launched
ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT / "data"

# The 21 topic columns, in the same order as in topic_model.py
TOPICS = [
    "Clean", "Comfort", "Facilities/Amenities", "Location", "Restaurant (dinner)",
    "Staff", "View (Balcony)", "Breakfast", "Room", "Pool", "Beach",
    "Bathroom/Shower (toilet)", "Bar", "Bed", "Parking", "Noise",
    "Reception-checkin", "Lift", "Value for money", "Wi-Fi", "Generic",
]

# ---------- Step 1: load and clean the dataset (same as topic_model.py) ----------

# Load the dataset (utf-8-sig removes the BOM at the start of the file)
df = pd.read_csv(DATA_FOLDER / "HRAST.csv", encoding="utf-8-sig")

# Remove the empty column without a name
df = df.drop(columns=["Unnamed: 2"])

# Convert the topic columns to integers: a strange value (e.g. "σ") becomes 0
for topic in TOPICS:
    df[topic] = pd.to_numeric(df[topic], errors="coerce").fillna(0).astype(int)

# Remove duplicate sentences (same text in "review"), keeping the first
df = df.drop_duplicates(subset="review")

# ---------- Step 2: train / validation / test split (same as topic_model.py) ----------

X = df["review"]
Y = df[TOPICS]

# First split: 15% of all sentences become the test set
X_rest, X_test, Y_rest, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)

# Second split: from the remaining 85%, take 15% of the TOTAL as validation (0.15 / 0.85 of the rest)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_rest, Y_rest, test_size=0.15 / 0.85, random_state=42
)

# Stop here if the sizes are not the ones of topic_model.py
assert (len(X_train), len(X_val), len(X_test)) == (16173, 3466, 3466), "The split is different!"

# ---------- Step 3: save the three files and check them ----------

parts = {"train": (X_train, Y_train), "val": (X_val, Y_val), "test": (X_test, Y_test)}

for name, (texts, labels) in parts.items():
    path = DATA_FOLDER / f"hrast_{name}.csv"
    table = pd.concat([texts, labels], axis=1)          # review + 21 topics, side by side
    table.to_csv(path, index=False, encoding="utf-8")

    # Read the file again: texts and topics must be identical to the ones in memory
    # (keep_default_na=False: a sentence like "None" must stay a text, not an empty value)
    saved = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    assert list(saved.columns) == ["review"] + TOPICS, "Wrong columns!"
    assert list(saved["review"]) == list(texts), "The sentences are different!"
    assert (saved[TOPICS].values == labels.values).all(), "The topics are different!"
    print(f"data/hrast_{name}.csv: {len(saved)} sentences, {len(saved.columns)} columns - checked")

# No sentence can be in two files
assert not (set(X_train) & set(X_val)), "Same sentence in train and validation!"
assert not (set(X_train) & set(X_test)), "Same sentence in train and test!"
assert not (set(X_val) & set(X_test)), "Same sentence in validation and test!"
print("Total:", len(X_train) + len(X_val) + len(X_test), "- no sentence in two files")
