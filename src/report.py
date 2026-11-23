import pandas as pd
from pipeline import (DATA_FOLDER, MODELS_FOLDER, TOPICS, sentiment_model, sentiment_vectorizer, predict_topics,
                      make_windows, topics_of_many_texts, clean_negative, clean_positive, split_full_text)

# Report "what to fix first" for ONE hotel.
# Importing this file does not run anything: the test at the end starts only with "python src/report.py".

N_TOP = 3          # how many topics get examples in "Fix first", and how many topics are in "Protect"
N_EXAMPLES = 2     # how many pieces of real text for each of them
MIN_WORDS = 4      # an example must have at least this number of words
MIN_REVIEWS = 30   # with fewer reviews than this, the report shows a warning
MIN_GAP = 0.05     # a topic enters "Fix first" only if the hotel exceeds the average by 5 percentage points...
MIN_CRITICS = 5    # ...and at least 5 reviews criticize the topic

# Weights removed from the reliable ones by hand. Neg_Bar: with 200 bootstrap resamples its interval
# is just below zero (-0.060, -0.001), but with 1,000 resamples it touches zero (-0.061, +0.002)
EXCLUDED_WEIGHTS = ["Neg_Bar"]

# Topics never shown in "Fix first" (they stay in the regression of Model 3). Generic: in our manual
# check of 50 reviews, 13 of the 19 criticisms counted as Generic were wrong
EXCLUDED_FROM_FIX_FIRST = ["Generic"]

# Names that a hotel manager understands
READABLE_NAMES = {
    "Clean": "Cleanliness",
    "Comfort": "Comfort",
    "Facilities/Amenities": "Facilities",
    "Location": "Location",
    "Restaurant (dinner)": "Restaurant",
    "Staff": "Staff",
    "View (Balcony)": "View",
    "Breakfast": "Breakfast",
    "Room": "Room",
    "Pool": "Pool",
    "Beach": "Beach",
    "Bathroom/Shower (toilet)": "Bathroom",
    "Bar": "Bar",
    "Bed": "Bed",
    "Parking": "Parking",
    "Noise": "Noise",
    "Reception-checkin": "Check-in & reception",
    "Lift": "Lift",
    "Value for money": "Value for money",
    "Wi-Fi": "Wi-Fi",
    "Generic": "Hotel overall (building, decor)",
}

def load_weights():
    """Read the file of Model 3. Return two dictionaries:
    - the reliable weights {column name: weight} (interval WITHOUT zero, not excluded by hand)
    - the average shares {column name: % of train reviews with that topic}, for ALL the columns."""
    table = pd.read_csv(MODELS_FOLDER / "impact_weights.csv", index_col="topic")
    contains_zero = (table["lower"] <= 0) & (table["upper"] >= 0)
    excluded = table.index.isin(EXCLUDED_WEIGHTS)
    weights = table.loc[~contains_zero & ~excluded, "coef"].to_dict()
    averages = table["share_pct"].to_dict()
    return weights, averages

def find_examples(texts, topics_per_text, topic, most_positive, used_windows):
    """Return up to N_EXAMPLES windows (at least MIN_WORDS words) where Model 1 found the topic.
    most_positive=False: the most NEGATIVE ones for the sentiment model; True: the most POSITIVE.
    At most one window per review. The sentiment chooses the examples, it never changes the counts.
    used_windows = windows already shown for other topics: they are not used again."""
    windows = []
    owner = []                                 # owner[k] = review that window k comes from
    for i, (text, topics) in enumerate(zip(texts, topics_per_text)):
        if topic not in topics:                # this review does not contain the topic
            continue
        for window in make_windows(text):
            if len(window.split()) >= MIN_WORDS:
                windows.append(window)
                owner.append(i)
    if len(windows) == 0:
        return []

    # Keep only the windows that triggered the topic, each with its P(positive)
    window_topics = predict_topics(windows)
    p_positive = sentiment_model.predict_proba(sentiment_vectorizer.transform(windows))[:, 1]
    candidates = []                            # (P(positive), window, review)
    for k in range(len(windows)):
        if topic in window_topics[k]:
            candidates.append((p_positive[k], windows[k], owner[k]))
    candidates.sort(key=lambda c: c[0], reverse=most_positive)

    examples = []
    used_reviews = []
    for _, window, review in candidates:
        if review not in used_reviews and window not in used_windows:
            examples.append(window)
            used_reviews.append(review)
            used_windows.append(window)        # the next topics will see it as already used
        if len(examples) == N_EXAMPLES:
            break
    return examples

def count_reviews(topics_per_text, topic):
    """Number of reviews that contain the topic."""
    return sum(1 for topics in topics_per_text if topic in topics)

def make_report(negative_texts, positive_texts):
    """Receive the negative and the positive texts of ONE hotel (one of each per review, "" if
    missing) and return the report: a dictionary with n_reviews, warning, fix_first and protect."""
    if len(negative_texts) != len(positive_texts) or len(negative_texts) == 0:
        raise ValueError("We need the same number (at least 1) of negative and positive texts")

    # Topics criticized and praised in each review (same steps used to build the Model 3 table)
    neg_texts = [clean_negative(t) for t in negative_texts]
    pos_texts = [clean_positive(t) for t in positive_texts]
    neg_topics = topics_of_many_texts(neg_texts)
    pos_topics = topics_of_many_texts(pos_texts)
    weights, averages = load_weights()

    n_reviews = len(negative_texts)
    used_windows = []                              # examples already shown, for all the topics

    # Fix first: only the topics criticized CLEARLY more than in the average hotel
    # (at least MIN_GAP above the average and at least MIN_CRITICS reviews).
    # points = share x Neg_ weight (all the points lost);
    # extra_points = (share - average share) x Neg_ weight (points lost compared with the average hotel)
    fix_first = []
    for topic in TOPICS:
        if topic in EXCLUDED_FROM_FIX_FIRST:
            continue
        n_critics = count_reviews(neg_topics, topic)
        share = n_critics / n_reviews
        average = averages["Neg_" + topic] / 100   # from % to a number from 0 to 1
        gap = share - average
        weight = weights.get("Neg_" + topic)       # None if the weight is not reliable
        clearly_above = gap >= MIN_GAP and n_critics >= MIN_CRITICS
        if weight is not None and weight < 0 and clearly_above:
            fix_first.append({"topic": topic, "name": READABLE_NAMES[topic],
                              "n_critics": n_critics, "share": share, "average": average,
                              "weight": weight, "points": share * weight,
                              "extra_points": gap * weight, "examples": []})
    fix_first.sort(key=lambda row: row["extra_points"])   # the most negative first
    for row in fix_first[:N_TOP]:
        row["examples"] = find_examples(neg_texts, neg_topics, row["topic"],
                                        most_positive=False, used_windows=used_windows)

    # Protect: points gained = share of reviews that praise the topic x its Pos_ weight
    # (only positive weights: Pos_Lift, Pos_Reception and Pos_Wi-Fi are reliable but negative)
    protect = []
    for topic in TOPICS:
        share = count_reviews(pos_topics, topic) / n_reviews
        weight = weights.get("Pos_" + topic)
        if weight is not None and weight > 0 and share > 0:
            protect.append({"topic": topic, "name": READABLE_NAMES[topic],
                            "share": share, "weight": weight,
                            "points": share * weight, "examples": []})
    protect.sort(key=lambda row: row["points"], reverse=True)   # the most positive first
    protect = protect[:N_TOP]
    for row in protect:
        row["examples"] = find_examples(pos_texts, pos_topics, row["topic"],
                                        most_positive=True, used_windows=used_windows)

    # Warning for hotels with few reviews ("" = no warning)
    warning = ""
    if n_reviews < MIN_REVIEWS:
        warning = "Few reviews: results are indicative only"

    return {"n_reviews": n_reviews, "warning": warning,
            "fix_first": fix_first, "protect": protect}

def make_report_from_full_texts(texts):
    """Like make_report, but each review is ONE text (Google, TripAdvisor...), not two Booking fields.
    The fixed rule + Model 2 (split_full_text in pipeline.py) rebuild the negative and the positive
    text of each review; then make_report does the rest. The report also says how many pieces
    were read and how many were put aside because Model 2 was not sure."""
    negative_texts, positive_texts = [], []
    n_pieces, n_aside = 0, 0
    for text in texts:
        negative, positive, pieces, aside = split_full_text(text)
        negative_texts.append(negative)
        positive_texts.append(positive)
        n_pieces += pieces
        n_aside += aside
    report = make_report(negative_texts, positive_texts)
    report["n_pieces"] = n_pieces
    report["n_aside"] = n_aside
    return report

def print_report(report):
    """Print the report returned by make_report (or by make_report_from_full_texts)."""
    if report["warning"] != "":
        print(report["warning"])
    print(f"Reviews read: {report['n_reviews']}")
    if "n_pieces" in report:               # only for reviews written as one text
        print(f"Pieces of text: {report['n_pieces']} - put aside (sentiment uncertain): {report['n_aside']}")
    print()
    print("FIX FIRST (topics clearly above the average of criticism)")
    if len(report["fix_first"]) == 0:
        print("     No topic is clearly worse than the average hotel")
    else:
        print(f"     {'Topic':<33}{'Criticized':>11}{'Average':>9}{'Weight':>9}{'vs average':>12}{'Total':>9}")
    for i, row in enumerate(report["fix_first"], start=1):
        print(f"{i:>3}. {row['name']:<33}{row['share'] * 100:>10.1f}%{row['average'] * 100:>8.1f}%"
              f"{row['weight']:>9.2f}{row['extra_points']:>12.3f}{row['points']:>9.3f}")
        for example in row["examples"]:
            print(f"       \"{example}\"")
    print()
    print("PROTECT")
    print(f"     {'Topic':<33}{'Praised':>11}{'Weight':>9}{'Points':>9}")
    for i, row in enumerate(report["protect"], start=1):
        print(f"{i:>3}. {row['name']:<33}{row['share'] * 100:>10.1f}%{row['weight']:>9.2f}{row['points']:>+9.3f}")
        for example in row["examples"]:
            print(f"       \"{example}\"")

# ---------- Test on 3 hotels of the Model 3 test set (only with "python src/report.py") ----------

if __name__ == "__main__":
    from sklearn.model_selection import GroupShuffleSplit

    # Load the reviews as in apply_topics.py: remove the duplicates, then the spaces at the borders
    df = pd.read_csv(DATA_FOLDER / "Hotel_Reviews.csv")
    df = df.drop_duplicates()
    for col in ["Hotel_Address", "Negative_Review", "Positive_Review"]:
        df[col] = df[col].str.strip()

    # Same split by hotel as in impact_full.py: these hotels were never seen by Model 3
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["Hotel_Address"]))
    test = df.iloc[test_idx]

    # True mean score of each test hotel, from the lowest to the highest
    hotel_scores = test.groupby("Hotel_Address")["Reviewer_Score"].mean().sort_values()
    median = hotel_scores.median()
    print(f"Test hotels: {len(hotel_scores)} - median of their mean scores: {median:.2f}")

    chosen = {
        "LOWEST mean score": hotel_scores.index[0],
        "CLOSEST TO THE MEDIAN": (hotel_scores - median).abs().idxmin(),
        "HIGHEST mean score": hotel_scores.index[-1],
    }
    for label, address in chosen.items():
        hotel = test[test["Hotel_Address"] == address]
        print()
        print("=" * 70)
        print(f"{label}: {hotel['Hotel_Name'].iloc[0]}")
        print(f"True mean score: {hotel_scores[address]:.2f}")
        print_report(make_report(list(hotel["Negative_Review"]), list(hotel["Positive_Review"])))
