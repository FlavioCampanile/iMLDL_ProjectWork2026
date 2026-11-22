# data/

This folder is **not** on GitHub (licenses and file size). Put these files here before running the scripts:

| File | Dataset | Where to get it |
|---|---|---|
| `HRAST.csv` | HRAST – hotel review sentences with topic and sentiment labels (CC BY-NC 4.0) | [Kaggle](https://www.kaggle.com/datasets/costastziouvas/hotel-reviews-aspects-sentiments-and-topics) |
| `Hotel_Reviews.csv` | 515K Hotel Reviews Data in Europe, Booking.com (CC0) | [Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) |
| `WWRSF_HotelReviewRegression.xlsx` | 300-review sample (only for `experiments/impact_model.py`) | shared by the team |

The folder name is lowercase (`data`): on Mac and Linux `Data` would be a different folder. Citation required for HRAST: see the main README.

Files created by the scripts: `review_topics.csv` (by `src/apply_topics.py`) and, optionally, `review_topics_nothing.csv`.
