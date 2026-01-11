# Recommendation-System

# MovieLens 100K Recommender (Matrix Factorization)

This project builds a simple movie recommendation system on the MovieLens 100K dataset using matrix factorization (SVD). It trains on historical ratings, evaluates on a held-out test set, and reports both RMSE and ranking metrics (Precision@10, NDCG@10).

## Dataset

Download the **MovieLens 100K** dataset from the official GroupLens website:

- https://grouplens.org/datasets/movielens/

Unzip it so that the `ml-100k` folder sits next to `ml_recommender.py`, for example:

```text
Recommendation System/
│
├─ ml_recommender.py
├─ README.md
└─ ml-100k/
   ├─ u.data
   └─ u.item
   ...
