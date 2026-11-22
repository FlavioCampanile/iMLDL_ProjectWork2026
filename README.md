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
│   ├── pipeline.py          shared functions to read a review (imported, not run)
│   ├── apply_topics.py      applies Model 1 to the 515K Booking.com reviews
│   ├── impact_full.py       Model 3 · impact on the rating (linear regression + bootstrap by hotel)
│   ├── report.py            report "what to fix first" for one hotel
│   ├── verify_hotels.py     predicted vs true score of the 224 test hotels
│   └── export_split.py      saves the train / validation / test sentences of Model 1 (for DistilBERT)
├── experiments/           earlier versions and sensitivity analyses (not part of the final pipeline)
│   ├── impact_model.py      first Model 3 on the 300-review sample (mid-term)
│   ├── impact_nothing.py    sensitivity analysis of the "nothing" rule
│   └── compare_distilbert.py  Model 1 (TF-IDF) vs DistilBERT on the same test sentences
├── notebooks/             Google Colab notebooks that train DistilBERT on the 21 topics (GPU needed)
├── results/               charts produced by the scripts
├── app/                   live demo (to be built)
├── docs/
│   ├── CODE_EXPLANATION.md  line-by-line guide to the code
│   ├── TRUSTWORTHY_AI.md    limits, risks, misuse, data bias (draft)
│   └── internal/            team notes (project context, to-do list)
├── data/                  datasets – NOT on GitHub (see data/README.md)
└── models/                trained models – NOT on GitHub (see models/README.md)
```

## Pipeline

| Step | Piece | What it does | Script |
|---|---|---|---|
| 1 | Model 1 · topics | Which of 21 topics a piece of a review mentions | `src/topic_model.py` |
| 2 | Model 2 · sentiment | Positive or negative, with a probability | `src/sentiment_model.py` |
| 3 | Topic table | Applies Model 1 to the 515K reviews: one 0/1 row per review | `src/apply_topics.py` |
| 4 | Model 3 · impact | Rating points lost or gained per topic | `src/impact_full.py` |
| 5 | Report | Ranks what to fix first and what to protect, for one hotel | `src/report.py` |

## How to reproduce the results

`data/` and `models/` are excluded from git, so after cloning you have to recreate them on your computer.

**1. Install the libraries** (results obtained with Python 3.12):

```
pip install -r requirements.txt
```

**2. Download the two datasets from Kaggle and put them in `data/`** (lowercase: on Mac and Linux `Data` would be a different folder):

- [HRAST](https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics) → `data/HRAST.csv`
- [515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) → `data/Hotel_Reviews.csv` (238 MB)

**3. Run the scripts in this order** (the commands are written for the repository root folder; the scripts find `data/` and `models/` by themselves):

| # | Command | Reads | Creates |
|---|---|---|---|
| 1 | `python src/topic_model.py` | `data/HRAST.csv` | `models/topic_model.joblib` |
| 2 | `python src/sentiment_model.py` | `data/HRAST.csv` | `models/sentiment_model.joblib` |
| 3 | `python src/apply_topics.py` | `data/Hotel_Reviews.csv` and the two `.joblib` files | `data/review_topics.csv` |
| 4 | `python src/impact_full.py` | `data/review_topics.csv` | `models/impact_weights.csv`, `models/impact_intercept.json` |
| 5 | `python src/report.py` | `data/Hotel_Reviews.csv`, `models/` | prints the report of 3 hotels never seen by Model 3 |
| 6 | `python src/verify_hotels.py` | `data/review_topics.csv`, `models/impact_weights.csv`, `models/impact_intercept.json` | `results/hotel_scores.png` (predicted vs true score of the 224 test hotels) |
| 7 | `python src/export_split.py` | `data/HRAST.csv` | `data/hrast_train.csv`, `data/hrast_val.csv`, `data/hrast_test.csv` (same split as script 1, for the DistilBERT experiment) |

`src/pipeline.py` contains the functions shared by the scripts (cleaning, windows, topics): it is imported and must not be run.

Scripts 1 and 2 can be swapped; 3 needs both of them, 4 needs the output of 3, 5 needs 1, 2 and 4, and 6 needs 3 and 4. 7 is independent of the others. On our computer script 3 took about 4 minutes and script 4 about 8-9 minutes (the bootstrap).

**Optional (sensitivity analysis):** set `USE_NOTHING_RULE = True` in `src/apply_topics.py`, run it again (it creates `data/review_topics_nothing.csv` and does not touch the original table), run `python experiments/impact_nothing.py`, then put the rule back to `False`.

**Optional (mid-term version of Model 3):** `python experiments/impact_model.py` reads `data/WWRSF_HotelReviewRegression.xlsx` (300-review sample).

## Datasets

| Dataset | Source | License | Used for |
|---|---|---|---|
| HRAST (23,113 review sentences, 21 topic labels + sentiment) | [Kaggle](https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics) | CC BY-NC 4.0 | Model 1 and Model 2 |
| 515K Hotel Reviews Data in Europe (Booking.com) | [Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe), by Jiashen Liu | CC0 | Model 3 |

We did not scrape any data ourselves (Booking.com terms, section A15.2). According to its Kaggle page, the 515K dataset was scraped from Booking.com by its author, and the data is originally owned by Booking.com.

### HRAST: attribution and citation

"Hotel Reviews: Aspects, Sentiments and Topics (HRAST)", published on [Kaggle](https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics) by Costas Tziouvas under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (attribution required, non-commercial use only). Changes we made: one empty column and 8 duplicate sentences are removed when the data is loaded for Model 1; Model 2 uses only part of the sentences. The dataset page lists these citations:

- Andreou, C., Tsapatsoulis, N., & Anastasopoulou, V. (2023). A Dataset of Hotel Reviews for Aspect-Based Sentiment Analysis and Topic Modeling. In *2023 18th International Workshop on Semantic and Social Media Adaptation & Personalization (SMAP 2023)* (pp. 1-9). IEEE. https://doi.org/10.1109/SMAP59435.2023.10255219
- Tsapatsoulis, N., Voutsa, M. C., & Djouvas, C. (2025). Biased by Design? Evaluating LLM Annotation Performance for Real-World and Synthetic Hotel Reviews. *AI*, forthcoming (as listed on the dataset page).

## Trustworthy AI

See [`docs/TRUSTWORTHY_AI.md`](docs/TRUSTWORTHY_AI.md).
