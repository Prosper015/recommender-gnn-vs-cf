import numpy as np
import pandas as pd
import torch

from src.models.lightgcn import (
    LightGCN,
    LightGCNRecommender,
    build_normalized_adjacency,
    sample_bpr_triplets,
)

# Petit graphe biparti synthetique, meme convention que graph_builder.py :
# 3 utilisateurs (noeuds 0,1,2), 4 items (noeuds 3,4,5,6).
# Aretes (deja symetriques, comme le produit build_bipartite_graph) :
#   user0-item0, user0-item1, user1-item2, user2-item3
NUM_USERS = 3
NUM_ITEMS = 4
EDGE_INDEX = np.array(
    [
        [0, 3, 0, 4, 1, 5, 2, 6],
        [3, 0, 4, 0, 5, 1, 6, 2],
    ]
)


def _make_model(num_layers=2, embedding_dim=8) -> LightGCN:
    model = LightGCN(NUM_USERS, NUM_ITEMS, embedding_dim=embedding_dim, num_layers=num_layers)
    model.set_graph(EDGE_INDEX)
    return model


def test_normalized_adjacency_simple_two_node_case():
    # user0 <-> item0 uniquement : degre(user0)=degre(item0)=1 -> valeur normalisee = 1
    edge_index = np.array([[0, 1], [1, 0]])
    adj = build_normalized_adjacency(edge_index, num_nodes=2).to_dense()
    assert torch.allclose(adj, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_propagate_output_shape_and_no_nan():
    model = _make_model()
    final = model.propagate()
    assert final.shape == (NUM_USERS + NUM_ITEMS, 8)
    assert not torch.isnan(final).any()


def test_get_user_item_scores_shape():
    model = _make_model()
    scores = model.get_user_item_scores(torch.tensor([0, 1]))
    assert scores.shape == (2, NUM_ITEMS)
    assert not torch.isnan(scores).any()


def test_sample_bpr_triplets_never_samples_a_known_positive():
    train_df = pd.DataFrame({"user_id": [0, 0, 1, 2], "item_id": [0, 1, 2, 3]})
    rng = np.random.default_rng(42)
    users, pos_items, neg_items = sample_bpr_triplets(train_df, num_items=NUM_ITEMS, rng=rng)

    positives = {0: {0, 1}, 1: {2}, 2: {3}}
    for u, neg in zip(users, neg_items):
        assert neg not in positives[int(u)]


def test_one_training_step_reduces_loss_and_stays_finite():
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    train_df = pd.DataFrame({"user_id": [0, 0, 1, 2], "item_id": [0, 1, 2, 3]})
    rng = np.random.default_rng(42)

    losses = []
    for _ in range(5):
        users, pos_items, neg_items = sample_bpr_triplets(train_df, num_items=NUM_ITEMS, rng=rng)
        optimizer.zero_grad()
        loss = model.bpr_loss(
            torch.from_numpy(users).long(),
            torch.from_numpy(pos_items).long(),
            torch.from_numpy(neg_items).long(),
        )
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_embedding_similarity_diagnostic_in_valid_cosine_range():
    model = _make_model()
    similarity = model.embedding_similarity_diagnostic(sample_size=50)
    assert -1.0 - 1e-6 <= similarity <= 1.0 + 1e-6


def test_lightgcn_recommender_score_items_matches_raw_model():
    model = _make_model()
    recommender = LightGCNRecommender(model)
    item_ids = np.array([0, 1, 2, 3])
    scores = recommender.score_items(user_id=0, item_ids=item_ids)

    expected = model.get_user_item_scores(torch.tensor([0])).squeeze(0).numpy()
    assert np.allclose(scores, expected)


def test_lightgcn_recommender_recommend_returns_top_n_sorted_desc():
    model = _make_model()
    recommender = LightGCNRecommender(model)
    recs = recommender.recommend(user_id=0, top_n=2)
    assert len(recs) == 2
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)
