# models/

The trained models are **on GitHub** (they are small), so the report and the demo work right after cloning, without retraining.

| File | What it is | Size | Created by |
|---|---|---|---|
| `topic_model.joblib` | Model 1 · topics (TF-IDF + 21 logistic regressions, one threshold per topic) | 1.2 MB | `src/topic_model.py` |
| `sentiment_model.joblib` | Model 2 · sentiment (TF-IDF + logistic regression) | 0.2 MB | `src/sentiment_model.py` |
| `impact_weights.csv`, `impact_intercept.json` | Model 3 · impact on the rating (42 weights with bootstrap interval, intercept) | 4 KB | `src/impact_full.py` |

The `.joblib` files were saved with scikit-learn 1.8.0 (see `requirements.txt`): load them with the same version. Running the scripts again recreates the same files (same numbers with the versions in `requirements.txt`).

DistilBERT (experiment on Model 1, 268 MB) is **not** here: it is trained with the notebooks in `notebooks/` on Google Colab.
