from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Compares Model 1 (TF-IDF + logistic regression) with DistilBERT on the same test sentences.
# DistilBERT is trained on Google Colab (notebooks/distilbert_topics*.ipynb): here we only
# choose its thresholds on the VALIDATION set and compute the F1 on the TEST set.

# Main folder of the repository (one level above the folder of this file):
# the paths work from whatever folder the script is launched
ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT / "data"
MODELS_FOLDER = ROOT / "models"

# Which training of DistilBERT: "distilbert" = 3 epochs, "distilbert8" = best of 8 epochs
RUN = "distilbert8"
print("DistilBERT run:", RUN)
print()

# ---------- Step 1: load the correct answers, Model 1 and the DistilBERT probabilities ----------

saved = joblib.load(MODELS_FOLDER / "topic_model.joblib")
TOPICS = saved["topics"]

val = pd.read_csv(DATA_FOLDER / "hrast_val.csv", keep_default_na=False)
test = pd.read_csv(DATA_FOLDER / "hrast_test.csv", keep_default_na=False)
bert_val = pd.read_csv(DATA_FOLDER / f"{RUN}_val_probs.csv", keep_default_na=False)
bert_test = pd.read_csv(DATA_FOLDER / f"{RUN}_test_probs.csv", keep_default_na=False)

# The rows must be the same sentences in the same order
assert list(val["review"]) == list(bert_val["review"]), "Validation sentences are different!"
assert list(test["review"]) == list(bert_test["review"]), "Test sentences are different!"

Y_val, Y_test = val[TOPICS].values, test[TOPICS].values
P_val, P_test = bert_val[TOPICS].values, bert_test[TOPICS].values

# ---------- Step 2: predictions of Model 1 (thresholds already saved in the model) ----------

X_test = saved["vectorizer"].transform(test["review"])
pred_tfidf = np.zeros(Y_test.shape, dtype=int)
for j, topic in enumerate(TOPICS):
    probs = saved["models"][topic].predict_proba(X_test)[:, 1]
    pred_tfidf[:, j] = probs >= saved["thresholds"][topic]

# ---------- Step 3: DistilBERT thresholds, chosen on VALIDATION (same rule as topic_model.py) ----------

candidates = np.round(np.arange(1, 19) * 0.05, 2)       # 0.05, 0.10, ..., 0.90
thresholds = np.zeros(len(TOPICS))
for j in range(len(TOPICS)):
    best_f1 = -1
    for t in candidates:
        f1 = f1_score(Y_val[:, j], (P_val[:, j] >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            thresholds[j] = t

pred_bert = (P_test >= thresholds).astype(int)

# ---------- Step 4: results on the TEST set ----------

def micro_macro(Y_true, Y_pred):
    """Micro F1 and macro F1, computed from the counts (fast, needed for the bootstrap)."""
    tp = ((Y_true == 1) & (Y_pred == 1)).sum(axis=0)      # correct "yes" of each topic
    fp = ((Y_true == 0) & (Y_pred == 1)).sum(axis=0)      # wrong "yes"
    fn = ((Y_true == 1) & (Y_pred == 0)).sum(axis=0)      # missed "yes"
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    macro = (2 * tp / np.maximum(2 * tp + fp + fn, 1)).mean()
    return micro, macro

f1_tfidf = f1_score(Y_test, pred_tfidf, average=None, zero_division=0)
f1_bert = f1_score(Y_test, pred_bert, average=None, zero_division=0)

print(f"{'Topic':<28}{'N test':>8}{'Thresh.':>9}{'TF-IDF':>9}{'BERT':>9}{'Diff.':>9}")
for j in np.argsort(f1_tfidf - f1_bert):                # from the biggest improvement
    print(f"{TOPICS[j]:<28}{Y_test[:, j].sum():>8}{thresholds[j]:>9.2f}"
          f"{f1_tfidf[j]:>9.4f}{f1_bert[j]:>9.4f}{f1_bert[j] - f1_tfidf[j]:>+9.4f}")

tfidf_scores, bert_scores = micro_macro(Y_test, pred_tfidf), micro_macro(Y_test, pred_bert)
print()
print(f"{'Model':<26}{'Micro F1':>10}{'Macro F1':>10}")
print(f"{'TF-IDF + LogReg tuned':<26}{tfidf_scores[0]:>10.4f}{tfidf_scores[1]:>10.4f}")
print(f"{'DistilBERT tuned':<26}{bert_scores[0]:>10.4f}{bert_scores[1]:>10.4f}")

# Total errors of the two models (one error = one wrong topic of one sentence)
for name, pred in [("TF-IDF", pred_tfidf), ("DistilBERT", pred_bert)]:
    extra, missed = ((pred == 1) & (Y_test == 0)).sum(), ((pred == 0) & (Y_test == 1)).sum()
    print(f"{name}: {extra} topics added by mistake, {missed} topics missed, "
          f"{(pred == Y_test).all(axis=1).mean():.1%} of the sentences completely correct")

# ---------- Step 5: is the difference bigger than chance? (bootstrap on the test sentences) ----------

# 1,000 times: draw 3,466 test sentences with replacement and compute the difference again.
# The SAME sentences are used for the two models, so the comparison is fair.
rng = np.random.RandomState(42)
differences = []
for _ in range(1000):
    rows = rng.randint(0, len(Y_test), len(Y_test))
    a = micro_macro(Y_test[rows], pred_tfidf[rows])
    b = micro_macro(Y_test[rows], pred_bert[rows])
    differences.append([b[0] - a[0], b[1] - a[1]])
low, high = np.percentile(differences, [2.5, 97.5], axis=0)

print()
for k, name in enumerate(["Micro", "Macro"]):
    print(f"{name} F1, DistilBERT - TF-IDF: {bert_scores[k] - tfidf_scores[k]:+.4f} "
          f"(95% interval from {low[k]:+.4f} to {high[k]:+.4f})")
