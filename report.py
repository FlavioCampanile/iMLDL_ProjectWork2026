import pandas as pd
from pipeline import (MODELS_FOLDER, TOPICS, models, thresholds, vectorizer,
                      make_windows, topics_of_many_texts, clean_negative, clean_positive)

# Report "what to fix first" for ONE hotel.
# Importing this file does not run anything: the test at the end starts only with "python report.py".

N_TOP = 3          # how many topics get examples in "Fix first", and how many topics are in "Protect"
N_EXAMPLES = 2     # how many pieces of real text for each of them

def load_weights():
    """Read the weights of Model 3 and keep only the reliable ones
    (the bootstrap interval does NOT contain zero). Return them as {column name: weight}."""
    weights = pd.read_csv(MODELS_FOLDER / "impact_weights.csv", index_col="topic")
    contains_zero = (weights["lower"] <= 0) & (weights["upper"] >= 0)
    return weights.loc[~contains_zero, "coef"].to_dict()

def find_examples(texts, topics_per_text, topic):
    """Return up to N_EXAMPLES windows where Model 1 found the topic: the most certain ones
    (highest probability), at most one per review."""
    best_windows = []                          # (probability, window), one per review
    for text, topics in zip(texts, topics_per_text):
        if topic not in topics:                # this review does not criticize the topic
            continue
        windows = make_windows(text)
        probs = models[topic].predict_proba(vectorizer.transform(windows))[:, 1]
        best = probs.argmax()                  # position of the most certain window
        if probs[best] >= thresholds[topic]:   # always true here: this window triggered the topic
            best_windows.append((probs[best], windows[best]))
    best_windows.sort(reverse=True)            # from the most certain to the least certain
    examples = []
    for _, window in best_windows:
        if window not in examples:             # two reviews can contain the same words
            examples.append(window)
        if len(examples) == N_EXAMPLES:
            break
    return examples

def share_of_reviews(topics_per_text, topic):
    """Share (from 0 to 1) of the reviews that contain the topic."""
    n_with_topic = sum(1 for topics in topics_per_text if topic in topics)
    return n_with_topic / len(topics_per_text)

def make_report(negative_texts, positive_texts):
    """Receive the negative and the positive texts of ONE hotel (one of each per review, "" if
    missing) and return the report: a dictionary with n_reviews, fix_first and protect."""
    if len(negative_texts) != len(positive_texts) or len(negative_texts) == 0:
        raise ValueError("We need the same number (at least 1) of negative and positive texts")

    # Topics criticized and praised in each review (same steps used to build the Model 3 table)
    neg_texts = [clean_negative(t) for t in negative_texts]
    pos_texts = [clean_positive(t) for t in positive_texts]
    neg_topics = topics_of_many_texts(neg_texts)
    pos_topics = topics_of_many_texts(pos_texts)
    weights = load_weights()

    # Fix first: points lost = share of reviews that criticize the topic x its Neg_ weight
    fix_first = []
    for topic in TOPICS:
        share = share_of_reviews(neg_topics, topic)
        weight = weights.get("Neg_" + topic)       # None if the weight is not reliable
        if weight is not None and weight < 0 and share > 0:
            fix_first.append({"topic": topic, "share": share, "weight": weight,
                              "points": share * weight, "examples": []})
    fix_first.sort(key=lambda row: row["points"])  # the most negative first
    for row in fix_first[:N_TOP]:
        row["examples"] = find_examples(neg_texts, neg_topics, row["topic"])

    # Protect: points gained = share of reviews that praise the topic x its Pos_ weight
    # (only positive weights: Pos_Lift, Pos_Reception and Pos_Wi-Fi are reliable but negative)
    protect = []
    for topic in TOPICS:
        share = share_of_reviews(pos_topics, topic)
        weight = weights.get("Pos_" + topic)
        if weight is not None and weight > 0 and share > 0:
            protect.append({"topic": topic, "share": share, "weight": weight,
                            "points": share * weight})
    protect.sort(key=lambda row: row["points"], reverse=True)   # the most positive first

    return {"n_reviews": len(negative_texts), "fix_first": fix_first, "protect": protect[:N_TOP]}

def print_report(report):
    """Print the report returned by make_report."""
    print(f"Reviews read: {report['n_reviews']}")
    print()
    print("FIX FIRST")
    print(f"     {'Topic':<28}{'Criticized':>11}{'Weight':>9}{'Points':>9}")
    for i, row in enumerate(report["fix_first"], start=1):
        print(f"{i:>3}. {row['topic']:<28}{row['share'] * 100:>10.1f}%{row['weight']:>9.2f}{row['points']:>9.3f}")
        for example in row["examples"]:
            print(f"       \"{example}\"")
    print()
    print("PROTECT")
    print(f"     {'Topic':<28}{'Praised':>11}{'Weight':>9}{'Points':>9}")
    for i, row in enumerate(report["protect"], start=1):
        print(f"{i:>3}. {row['topic']:<28}{row['share'] * 100:>10.1f}%{row['weight']:>9.2f}{row['points']:>+9.3f}")

# ---------- Test on 3 hotels of the Model 3 test set (only with "python report.py") ----------

if __name__ == "__main__":
    from sklearn.model_selection import GroupShuffleSplit

    # Load the reviews as in apply_topics.py: remove the duplicates, then the spaces at the borders
    df = pd.read_csv("Data/Hotel_Reviews.csv")
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
