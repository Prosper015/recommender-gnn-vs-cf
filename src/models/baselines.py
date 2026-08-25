"""
Baselines classiques de filtrage collaboratif : SVD et Item-Item CF, via
scikit-surprise. Entrainees directement sur les IDs BRUTS MovieLens (pas de
reindexation necessaire : contrairement a LightGCN, scikit-surprise gere des
identifiants arbitraires en interne) -- ce qui les rend directement
compatibles avec le contrat backend, qui appelle predict(user_id, movie_id)
avec des movieId bruts (voir feature/fastapi-backend: src/services/model_service.py).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader

from src.models.common import Recommender


def _build_trainset(train_df: pd.DataFrame):
    reader = Reader(rating_scale=(float(train_df["rating"].min()), float(train_df["rating"].max())))
    dataset = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
    return dataset.build_full_trainset()


class SVDRecommender(Recommender):
    """Factorisation matricielle (Koren et al., 2009), implementation scikit-surprise."""

    def __init__(self, n_factors: int = 100, n_epochs: int = 20, random_state: int = 42):
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
        self.all_item_ids_: np.ndarray = np.array([], dtype=np.int64)

    def fit(self, train_df: pd.DataFrame) -> "SVDRecommender":
        self.model.fit(_build_trainset(train_df))
        self.all_item_ids_ = train_df["item_id"].unique().astype(np.int64)
        return self

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        return np.array([self.model.predict(int(user_id), int(i)).est for i in item_ids])


class ItemItemCFRecommender(Recommender):
    """Filtrage collaboratif item-item (k plus proches voisins, similarite cosinus)."""

    def __init__(self, k_neighbors: int = 40):
        self.model = KNNBasic(
            k=k_neighbors,
            sim_options={"name": "cosine", "user_based": False},
            verbose=False,
        )
        self.all_item_ids_: np.ndarray = np.array([], dtype=np.int64)

    def fit(self, train_df: pd.DataFrame) -> "ItemItemCFRecommender":
        self.model.fit(_build_trainset(train_df))
        self.all_item_ids_ = train_df["item_id"].unique().astype(np.int64)
        return self

    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        return np.array([self.model.predict(int(user_id), int(i)).est for i in item_ids])

    def precompute_recommendations(
        self, user_ids: Iterable[int], top_n: int = 10
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Precalcule le Top-N par utilisateur sous forme de dict brut
        {user_id: [(item_id, score), ...]}. C'est ce format (et pas un objet
        avec une methode .recommend()) que model_service.py sait consommer
        nativement pour l'item-item CF (branche `isinstance(model, dict)`).
        """
        return {int(u): self.recommend(u, top_n=top_n) for u in user_ids}


def validation_rmse(recommender: SVDRecommender | ItemItemCFRecommender, val_df: pd.DataFrame) -> float:
    """
    RMSE sur validation (erreur de prediction de note, pas de ranking).

    Sert a tracer une courbe de convergence/sensibilite (RMSE vs
    hyperparametre) : contrairement a evaluate_full_ranking (~1.5M
    predictions sur 100k, lent), ceci ne fait qu'une prediction par ligne de
    validation (943 lignes) -- assez rapide pour reentrainer plusieurs
    configurations d'affilee. Fonctionne pour SVD ET Item-Item CF : les deux
    exposent le meme .model.test() de scikit-surprise.
    """
    from surprise import accuracy

    testset = list(zip(val_df["user_id"], val_df["item_id"], val_df["rating"]))
    predictions = recommender.model.test(testset)
    return accuracy.rmse(predictions, verbose=False)


def save_svd(recommender: SVDRecommender, out_path: Path) -> None:
    """
    Persiste l'algorithme scikit-surprise BRUT (recommender.model), pas le
    wrapper : c'est exactement l'objet que model_service.py recharge et sur
    lequel il appelle .predict(user_id, movie_id) directement.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(recommender.model, f)


def save_item_item(
    recommender: ItemItemCFRecommender,
    user_ids: Iterable[int],
    out_path: Path,
    top_n: int = 10,
) -> None:
    """Persiste le dict precalcule {user_id: [(item_id, score), ...]} attendu par model_service.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    precomputed = recommender.precompute_recommendations(user_ids, top_n=top_n)
    with open(out_path, "wb") as f:
        pickle.dump(precomputed, f)
