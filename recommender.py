import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split


# Load user-item ratings: user_id, item_id, rating, timestamp.
def load_ratings(path="ml-100k/u.data"):
    cols = ["user_id", "item_id", "rating", "timestamp"]
    ratings = pd.read_csv(path, sep="\t", names=cols, engine="python")
    return ratings


# Load movie IDs and titles from u.item.
def load_items(path="ml-100k/u.item"):
    item_cols = ["item_id", "title"] + [f"col_{i}" for i in range(22)]
    items = pd.read_csv(path, sep="|", names=item_cols, encoding="latin-1", engine="python")
    items = items[["item_id", "title"]]
    return items


# Train an SVD matrix-factorization model using Surprise.
def train_svd_model(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "item_id", "rating"]], reader)
    trainset, _ = surprise_split(data, test_size=0.2, random_state=42)

    algo = SVD(n_factors=50, n_epochs=20, random_state=42)
    algo.fit(trainset)
    return algo


# Compute RMSE of the model on a held-out test set.
def compute_rmse(algo, test_ratings):
    testset = [
        (int(uid), int(iid), float(r))
        for uid, iid, r in test_ratings[["user_id", "item_id", "rating"]].values
    ]
    predictions = algo.test(testset)

    mse = np.mean([(pred.est - pred.r_ui) ** 2 for pred in predictions])
    rmse = np.sqrt(mse)
    return rmse


# Precision@k for a single user.
def precision_at_k(group, k=10):
    recs = group["recommended_items"].iloc[0][:k]
    # Consider items with rating >= 4 as "relevant"
    relevant = set(group[group["rating"] >= 4]["item_id"])
    if not recs:
        return 0.0
    hits = len(set(recs) & relevant)
    return hits / float(k)


# NDCG@k for a single user.
def ndcg_at_k(group, k=10):
    recs = group["recommended_items"].iloc[0][:k]
    relevant = set(group[group["rating"] >= 4]["item_id"])

    dcg = 0.0
    idcg = 0.0

    for i, item_id in enumerate(recs):
        rel = 1.0 if item_id in relevant else 0.0
        dcg += rel / np.log2(i + 2)  

    ideal_rels = [1.0] * min(len(relevant), k)
    for i, rel in enumerate(ideal_rels):
        idcg += rel / np.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# Compute Precision@k and NDCG@k.
def evaluate_ranking(algo, ratings, train_ratings, test_ratings, num_users=100, k=10):
    all_items = set(ratings["item_id"].unique())
    test_users = test_ratings["user_id"].unique()[:num_users]

    rows = []

    for user_id in test_users:
        user_id_int = int(user_id)

        user_test = test_ratings[test_ratings["user_id"] == user_id_int]
        if user_test.empty:
            continue

        user_train_items = set(
            train_ratings[train_ratings["user_id"] == user_id_int]["item_id"].unique()
        )

        candidates = list(all_items - user_train_items)
        if not candidates:
            continue

        candidates_subset = candidates[:500]
        scores = []
        for item_id in candidates_subset:
            pred = algo.predict(user_id_int, int(item_id))
            scores.append((int(item_id), pred.est))

        scores.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, _ in scores[:50]]

        user_rows = user_test.copy()
        user_rows["recommended_items"] = [recommended_items] * len(user_rows)
        rows.append(user_rows)

    if not rows:
        return 0.0, 0.0

    all_results = pd.concat(rows, ignore_index=True)

    precisions = all_results.groupby("user_id").apply(precision_at_k, k=k, include_groups=False)
    ndcgs = all_results.groupby("user_id").apply(ndcg_at_k, k=k, include_groups=False)

    return precisions.mean(), ndcgs.mean()


# Print top-N recommended movies for a given user with titles.
def show_example_recommendations(algo, items, ratings, train_ratings, user_id=1, top_n=10):
    user_id_int = int(user_id)

    all_items = set(ratings["item_id"].unique())
    user_train_items = set(
        train_ratings[train_ratings["user_id"] == user_id_int]["item_id"].unique()
    )
    candidates = list(all_items - user_train_items)

    if not candidates:
        print(f"No candidate items for user {user_id_int}.")
        return

    candidates_subset = candidates[:500]
    scores = []
    for item_id in candidates_subset:
        pred = algo.predict(user_id_int, int(item_id))
        scores.append((int(item_id), pred.est))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_recs = scores[:top_n]

    print(f"\nTop {top_n} recommendations for user {user_id_int}:")
    print("Movie ID | Predicted Rating | Title")
    for item_id, score in top_recs:
        title_row = items[items["item_id"] == item_id]
        if title_row.empty:
            title = "Unknown title"
        else:
            title = title_row["title"].iloc[0]
        print(f"{item_id:7d} | {score:6.2f}           | {title}")


def main():
    print("Loading data...")
    ratings = load_ratings()
    items = load_items()

    print(
        f"Loaded {len(ratings)} ratings from "
        f"{ratings['user_id'].nunique()} users and {ratings['item_id'].nunique()} movies."
    )

    print("\nSplitting into train/test...")
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=0.2, random_state=42
    )

    print("\nTraining SVD (matrix factorization) model...")
    algo = train_svd_model(train_ratings)

    print("\nEvaluating RMSE on test set...")
    rmse = compute_rmse(algo, test_ratings)
    print(f"Test RMSE: {rmse:.4f}")

    print("\nEvaluating ranking metrics (Precision@10, NDCG@10)...")
    precision, ndcg = evaluate_ranking(
        algo, ratings, train_ratings, test_ratings, num_users=100, k=10
    )
    print(f"Mean Precision@10: {precision:.4f}")
    print(f"Mean NDCG@10:      {ndcg:.4f}")

    print("\nExample recommendations:")
    show_example_recommendations(algo, items, ratings, train_ratings, user_id=1, top_n=10)


if __name__ == "__main__":
    main()
