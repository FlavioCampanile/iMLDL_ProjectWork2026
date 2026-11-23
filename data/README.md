# data/

Only part of this folder is on GitHub. The folder name is lowercase (`data`): on Mac and Linux `Data` would be a different folder.

## On GitHub

| File | What it is |
|---|---|
| `Hotel_Database.xlsx` | 40 Swiss hotels catalogued by the team (city, cluster, Booking stars, restaurant, wellness, parking, ski): candidates for the Swiss test set |

## To download from Kaggle (not on GitHub: licenses and file size)

| File | Dataset | Where to get it |
|---|---|---|
| `HRAST.csv` | HRAST – hotel review sentences with topic and sentiment labels (CC BY-NC 4.0) | [Kaggle](https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics) |
| `Hotel_Reviews.csv` | 515K Hotel Reviews Data in Europe, Booking.com (CC0), 238 MB | [Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) |

Citation required for HRAST: see the main README.

## Created by the scripts (not on GitHub: run the scripts to recreate them)

- `review_topics.csv` (77 MB) by `src/apply_topics.py`; optionally `review_topics_nothing.csv`
- `hrast_train.csv`, `hrast_val.csv`, `hrast_test.csv` by `src/export_split.py` (the split of Model 1)
- `manual_check.xlsx` by `src/check_pieces.py`

## Kept by the team, not on GitHub (they contain reviews copied by hand)

These files contain review texts written by real hotel guests, copied by hand from public review pages (without the guests' names). They are shared within the team only.

| File | What it is | Used by |
|---|---|---|
| `citizenm_zurich_booking.csv` | 30 Booking.com reviews of citizenM Zürich (liked / disliked blocks and score) | `src/check_booking_hotel.py` |
| `google_check.xlsx` | 30 reviews copied by hand as one text each, with score | `src/check_pieces.py`, `src/explain_hotel.py` |
| `manual_check.xlsx`, `manual_check_CLAUDE_labels.xlsx` | manual check of the fixed rule (empty / pre-labelled by an AI model, to be reviewed by us) | summary in `results/manual_check_summary.md` |
| `WWRSF_HotelReviewRegression.xlsx` | 300-review sample of the 515K (only for `experiments/impact_model.py`) | `experiments/impact_model.py` |
