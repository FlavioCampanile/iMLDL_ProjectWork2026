# Roomor

> *Your guests talk. We translate.*

Roomor reads a hotel's guest reviews and tells the manager **what to fix first**, ranking topics by how many rating points they cost (and which strengths to protect).

**Course:** 7,230 | 8,230 - Introduction to Machine Learning and Deep Learning (Autumn Semester 2026)  
**Institution:** University of St. Gallen (HSG)  
**Founders (Team 1):** Flavio Massimo Campanile & Filippo Strafile  
**Instructor:** Prof. Dr. Bruno Rodrigues

---

## Repository structure

```
iMLDL_ProjectWork2026/
├── README.md              this file
├── requirements.txt       Python libraries and versions
├── src/                   final pipeline (the code that builds the product)
│   ├── topic_model.py       Model 1 · topics (21 logistic regressions on TF-IDF)
│   ├── sentiment_model.py   Model 2 · sentiment (logistic regression on TF-IDF)
│   ├── apply_topics.py      applies Model 1 to the 515K Booking.com reviews
│   └── impact_full.py       Model 3 · impact on the rating (linear regression + bootstrap by hotel)
├── experiments/           earlier versions and sensitivity analyses (not part of the final pipeline)
│   ├── impact_model.py      first Model 3 on the 300-review sample (mid-term)
│   └── impact_nothing.py    sensitivity analysis of the "nothing" rule
├── app/                   live demo (to be built)
├── docs/
│   ├── CODE_EXPLANATION.md  line-by-line guide to the code
│   ├── TRUSTWORTHY_AI.md    limits, risks, misuse, data bias (draft)
│   └── internal/            team notes (project context, to-do list)
├── Data/                  datasets – NOT on GitHub (see Data/README.md)
└── models/                trained models – NOT on GitHub (see models/README.md)
```

## Pipeline

| Step | Piece | What it does | Script |
|---|---|---|---|
| 1 | Model 1 · topics | Which of 21 topics a piece of a review mentions | `src/topic_model.py` |
| 2 | Model 2 · sentiment | Positive or negative, with a probability | `src/sentiment_model.py` |
| 3 | Topic table | Applies Model 1 to the 515K reviews: one 0/1 row per review | `src/apply_topics.py` |
| 4 | Model 3 · impact | Rating points lost or gained per topic | `src/impact_full.py` |

## How to reproduce the results

`Data/` and `models/` are excluded from git, so after cloning you have to recreate them on your computer.

**1. Install the libraries** (results obtained with Python 3.12):

```
pip install -r requirements.txt
```

**2. Put the input datasets in `Data/`** (download links in `Data/README.md`): `HRAST.csv` and `Hotel_Reviews.csv`.

**3. Run the scripts from the repository root folder (not from inside `src/`), in this order:**

| # | Command | Reads | Creates |
|---|---|---|---|
| 1 | `python src/topic_model.py` | `Data/HRAST.csv` | `models/topic_model.joblib` |
| 2 | `python src/sentiment_model.py` | `Data/HRAST.csv` | `models/sentiment_model.joblib` |
| 3 | `python src/apply_topics.py` | `Data/Hotel_Reviews.csv` and the two `.joblib` files | `Data/review_topics.csv` |
| 4 | `python src/impact_full.py` | `Data/review_topics.csv` | `models/impact_weights.csv`, `models/impact_intercept.json` |

Scripts 1 and 2 can be swapped; 3 needs both of them, and 4 needs the output of 3. On our computer script 3 took about 4 minutes and script 4 about 8-9 minutes (the bootstrap).

**Optional (sensitivity analysis):** set `USE_NOTHING_RULE = True` in `src/apply_topics.py`, run it again (it creates `Data/review_topics_nothing.csv` and does not touch the original table), run `python experiments/impact_nothing.py`, then put the rule back to `False`.

**Optional (mid-term version of Model 3):** `python experiments/impact_model.py` reads `Data/WWRSF_HotelReviewRegression.xlsx` (300-review sample).

## Data

| Dataset | Source | License | Used for |
|---|---|---|---|
| HRAST (23,113 review sentences, 21 topic labels + sentiment) | Kaggle | CC BY-NC 4.0 | Model 1 and Model 2 |
| 515K Hotel Reviews Data in Europe (Booking.com) | Kaggle | CC0 | Model 3 |

No data was scraped (Booking.com terms, section A15.2).

## Trustworthy AI

See [`docs/TRUSTWORTHY_AI.md`](docs/TRUSTWORTHY_AI.md).
