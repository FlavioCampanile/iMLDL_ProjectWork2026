# INSERT THE NAME OF THE STARTUP

**Course:** 7,230 | 8,230 - Introduction to Machine Learning and Deep Learning (Autumn Semester 2026)  
**Institution:** University of St. Gallen (HSG)  
**Founders (Team 1):** Flavio Massimo Campanile & Filippo Strafile  
**Instructor:** Prof. Dr. Bruno Rodrigues  

---

## How to recreate the `models/` folder (draft)

`Data/` and `models/` are excluded from git, so after cloning you have to recreate them on your computer.

**1. Install the libraries** (the results were obtained with Python 3.12, scikit-learn 1.8.0, pandas 3.0.1, numpy 2.2.6, joblib 1.5.3):

```
pip install scikit-learn pandas numpy joblib
```

**2. Put the input datasets in `Data/`** (they are shared outside GitHub): `HRAST.csv` and `Hotel_Reviews.csv`.

**3. Run the scripts from the repository folder, in this order:**

| # | Command | Reads | Creates |
|---|---|---|---|
| 1 | `python topic_model.py` | `Data/HRAST.csv` | `models/topic_model.joblib` |
| 2 | `python sentiment_model.py` | `Data/HRAST.csv` | `models/sentiment_model.joblib` |
| 3 | `python apply_topics.py` | `Data/Hotel_Reviews.csv` and the two `.joblib` files | `Data/review_topics.csv` |
| 4 | `python impact_full.py` | `Data/review_topics.csv` | `models/impact_weights.csv`, `models/impact_intercept.json` |

`pipeline.py` contains the functions shared by the scripts (cleaning, windows, topics): it is imported and must not be run.

Scripts 1 and 2 can be swapped; 3 needs both of them, and 4 needs the output of 3. On our computer script 3 took about 4 minutes and script 4 about 8-9 minutes (the bootstrap).

**Optional:** to compare the results with the "nothing" rule, set `USE_NOTHING_RULE = True` in `apply_topics.py`, run it again (it creates `Data/review_topics_nothing.csv` and does not touch the original table), run `python impact_nothing.py`, then put the rule back to `False`.
