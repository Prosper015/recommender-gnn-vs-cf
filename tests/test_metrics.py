import math

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    build_seen_items_per_user,
    evaluate_full_ranking,
    mlflow_safe_names,
    ndcg_at_k,
    precision_at_k,
    rank_of_item,
    recall_at_k,
)


def test_rank_of_item_basic_ordering():
    scores = np.array([1.0, 5.0, 3.0])
    candidates = np.array([10, 20, 30])
    assert rank_of_item(scores, candidates, target_item=20) == 1
    assert rank_of_item(scores, candidates, target_item=30) == 2
    assert rank_of_item(scores, candidates, target_item=10) == 3


def test_rank_of_item_ties_broken_by_lower_item_id():
    scores = np.array([5.0, 5.0, 3.0])
    candidates = np.array([10, 20, 30])
    assert rank_of_item(scores, candidates, target_item=10) == 1
    assert rank_of_item(scores, candidates, target_item=20) == 2


def test_rank_of_item_missing_target_returns_none():
    scores = np.array([1.0, 2.0])
    candidates = np.array([10, 20])
    assert rank_of_item(scores, candidates, target_item=99) is None


def test_precision_recall_ndcg_hit_inside_k():
    rank = 3
    assert precision_at_k(rank, k=5) == 1 / 5
    assert recall_at_k(rank, k=5) == 1.0
    assert ndcg_at_k(rank, k=5) == 1.0 / math.log2(4)


def test_precision_recall_ndcg_miss_outside_k():
    rank = 6
    assert precision_at_k(rank, k=5) == 0.0
    assert recall_at_k(rank, k=5) == 0.0
    assert ndcg_at_k(rank, k=5) == 0.0


def test_precision_recall_ndcg_none_rank_is_a_miss():
    assert precision_at_k(None, k=5) == 0.0
    assert recall_at_k(None, k=5) == 0.0
    assert ndcg_at_k(None, k=5) == 0.0


def test_build_seen_items_per_user_merges_multiple_dfs():
    train_df = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [1, 2, 3]})
    val_df = pd.DataFrame({"user_id": [0, 1], "item_id": [5, 4]})
    seen = build_seen_items_per_user(train_df, val_df)
    assert seen[0] == {1, 2, 5}
    assert seen[1] == {3, 4}


def test_evaluate_full_ranking_matches_hand_computed_averages():
    # user 0 : deja vu item 0, cible = item 1, candidats = [1, 2, 3]
    # user 1 : rien de vu, cible = item 3, candidats = [0, 1, 2, 3]
    test_df = pd.DataFrame({"user_id": [0, 1], "item_id": [1, 3]})
    seen_items_per_user = {0: {0}, 1: set()}
    all_item_ids = np.arange(4)

    user0_scores = {1: 3.0, 2: 2.0, 3: 1.0}  # item 1 est classe 1er (rank=1)
    user1_scores = {0: 1.0, 1: 5.0, 2: 2.0, 3: 4.0}  # item 3 est classe 2eme (rank=2)

    def score_fn(user_id, candidate_items):
        table = user0_scores if user_id == 0 else user1_scores
        return np.array([table[i] for i in candidate_items])

    metrics = evaluate_full_ranking(
        score_fn, test_df, seen_items_per_user, all_item_ids=all_item_ids, k_list=[1, 3]
    )

    assert metrics["precision@1"] == 0.5
    assert metrics["recall@1"] == 0.5
    assert metrics["ndcg@1"] == 0.5

    assert math.isclose(metrics["precision@3"], 1 / 3)
    assert metrics["recall@3"] == 1.0
    expected_ndcg3 = (1.0 + 1.0 / math.log2(3)) / 2
    assert math.isclose(metrics["ndcg@3"], expected_ndcg3)


def test_mlflow_safe_names_strips_at_symbol():
    metrics = {"precision@5": 0.1, "ndcg@10": 0.2, "embedding_cosine_similarity": 0.3}
    safe = mlflow_safe_names(metrics)
    assert safe == {
        "precision_at_5": 0.1,
        "ndcg_at_10": 0.2,
        "embedding_cosine_similarity": 0.3,
    }


def test_evaluate_full_ranking_raises_on_empty_test_set():
    test_df = pd.DataFrame({"user_id": [], "item_id": []})
    try:
        evaluate_full_ranking(lambda u, c: np.zeros(len(c)), test_df, {}, all_item_ids=np.arange(4))
    except ValueError:
        return
    raise AssertionError("evaluate_full_ranking aurait du lever ValueError sur un test_df vide")
