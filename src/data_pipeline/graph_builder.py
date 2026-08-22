"""
Construction du graphe biparti Utilisateur-Item.

Convention d'indexation des noeuds :
    - noeuds [0, num_users - 1]                    -> utilisateurs
    - noeuds [num_users, num_users + num_items - 1] -> items
Necessaire car LightGCN utilise un seul espace d'indices pour tous les
noeuds (contrairement a un graphe heterogene classique).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BipartiteGraph:
    """Representation generique du graphe biparti, sans dependance a PyG/DGL."""

    edge_index: np.ndarray   # shape (2, num_edges), deja symetrique (u->i et i->u)
    num_users: int
    num_items: int
    num_nodes: int           # = num_users + num_items
    edge_weight: np.ndarray | None = None

    def summary(self) -> dict:
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_nodes": self.num_nodes,
            "num_edges_directed": int(self.edge_index.shape[1]),
            "num_interactions": int(self.edge_index.shape[1] // 2),
            "avg_degree": round(self.edge_index.shape[1] / self.num_nodes, 2),
        }


def build_bipartite_graph(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    use_rating_as_weight: bool = False,
) -> BipartiteGraph:
    """
    Construit le graphe biparti a partir du TRAIN set uniquement.

    Regle stricte : on construit le graphe UNIQUEMENT sur train_df, jamais
    sur val/test, sinon le modele "verrait" par la structure du graphe les
    aretes qu'il doit predire (meme logique anti-leakage que pour le split).
    """
    _validate_ids_in_range(train_df, num_users, num_items)

    user_nodes = train_df["user_id"].to_numpy()
    item_nodes = train_df["item_id"].to_numpy() + num_users  # decalage -> espace unifie

    src = np.concatenate([user_nodes, item_nodes])
    dst = np.concatenate([item_nodes, user_nodes])
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    edge_weight = None
    if use_rating_as_weight:
        ratings = train_df["rating"].to_numpy().astype(np.float32)
        edge_weight = np.concatenate([ratings, ratings])

    graph = BipartiteGraph(
        edge_index=edge_index,
        num_users=num_users,
        num_items=num_items,
        num_nodes=num_users + num_items,
        edge_weight=edge_weight,
    )
    logger.info("Graphe biparti construit : %s", graph.summary())
    return graph


def _validate_ids_in_range(df: pd.DataFrame, num_users: int, num_items: int) -> None:
    assert df["user_id"].min() >= 0 and df["user_id"].max() < num_users, (
        "user_id hors de l'intervalle [0, num_users). Le DataFrame doit avoir "
        "ete reindexe (voir download.reindex_ids) avant de construire le graphe."
    )
    assert df["item_id"].min() >= 0 and df["item_id"].max() < num_items, (
        "item_id hors de l'intervalle [0, num_items). Le DataFrame doit avoir "
        "ete reindexe (voir download.reindex_ids) avant de construire le graphe."
    )


def save_graph(graph: BipartiteGraph, out_path) -> None:
    """Sauvegarde le graphe generique au format .npz (portable, sans dependance)."""
    np.savez(
        out_path,
        edge_index=graph.edge_index,
        num_users=graph.num_users,
        num_items=graph.num_items,
        num_nodes=graph.num_nodes,
        edge_weight=graph.edge_weight if graph.edge_weight is not None else np.array([]),
    )
    logger.info("Graphe sauvegarde : %s", out_path)