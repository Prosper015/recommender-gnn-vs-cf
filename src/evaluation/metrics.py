"""
Metriques de ranking pour l'evaluation offline (Precision@K, Recall@K, NDCG@K).

Protocole : full-ranking leave-one-out, comme dans He et al. "LightGCN" (2020)
et Rendle/Koren NCF. Pour chaque utilisateur, le split temporel (voir
src/data_pipeline/temporal_split.py) laisse EXACTEMENT un item pertinent en
test. On classe cet item parmi TOUS les items non vus en train/val (pas
d'echantillonnage de negatifs : Krichene & Rendle, 2020, montrent que les
metriques sur negatifs echantillonnes peuvent etre trompeuses), et on mesure
a quel rang il ressort.

Avec un seul item pertinent par utilisateur :
    hit@k     = 1 si le rang du bon item <= k, sinon 0
    recall@k  = hit@k                          (il n'y a qu'1 seul pertinent)
    precision@k = hit@k / k
    ndcg@k    = 1 / log2(rank + 1) si hit, sinon 0   (IDCG = 1 avec 1 pertinent)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np

DEFAULT_K_LIST = [5, 10, 20]


def rank_of_item(scores: np.ndarray, candidate_items: np.ndarray, target_item: int) -> int | None:
    """
    Classe `target_item` parmi `candidate_items` selon `scores` (memes tailles,
    scores[i] correspond a candidate_items[i]). Retourne le rang 1-indexe du
    target (1 = meilleur score), ou None si target_item n'est pas dans candidate_items.

    Les egalites de score sont departagees par ordre croissant d'item_id
    (deterministe, reproductible) plutot que par l'ordre d'iteration.
    """
    idx = np.where(candidate_items == target_item)[0]
    if idx.size == 0:
        return None
    target_score = scores[idx[0]]

    n_strictly_better = int(np.sum(scores > target_score))
    tied = candidate_items[scores == target_score]
    n_tied_before = int(np.sum(tied < target_item))
    return n_strictly_better + n_tied_before + 1


def hit_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def precision_at_k(rank: int | None, k: int) -> float:
    return hit_at_k(rank, k) / k


def recall_at_k(rank: int | None, k: int) -> float:
    # Cas LOO strict : un seul item pertinent -> recall@k == hit@k
    return hit_at_k(rank, k)


def ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate_full_ranking(
    score_fn: Callable[[int, np.ndarray], np.ndarray],
    test_df,
    seen_items_per_user: dict[int, set[int]],
    all_item_ids: np.ndarray | Iterable[int],
    k_list: Iterable[int] = DEFAULT_K_LIST,
) -> dict[str, float]:
    """
    Evalue un modele sur le protocole full-ranking LOO.

    Parametres
    ----------
    score_fn : (user_id, candidate_items) -> scores
        Doit retourner un score par item de `candidate_items`, plus haut =
        plus pertinent. Les 3 methodes (SVD, Item-Item CF, LightGCN) exposent
        toutes une fonction compatible via leur wrapper `Recommender`.
    test_df : DataFrame avec colonnes [user_id, item_id, ...] - exactement
        une ligne par utilisateur evalue (sortie de leave_one_out_split).
    seen_items_per_user : dict user_id -> set d'item_id deja vus en train+val,
        exclus des candidats pour eviter de "recommander" ce que l'utilisateur
        a deja consomme.
    all_item_ids : univers complet des item_id candidats. NE PAS supposer
        [0, N) : les baselines (SVD/Item-Item CF) travaillent sur les
        movieId BRUTS MovieLens (non contigus), tandis que LightGCN travaille
        sur des indices internes reindexes [0, num_items). Chaque appelant
        passe l'espace qui correspond a sa propre methode.

    Retourne un dict aplati {"precision@5": ..., "recall@10": ..., "ndcg@20": ...}
    moyenne sur tous les utilisateurs du test.
    """
    k_list = list(k_list)
    all_items = np.asarray(list(all_item_ids))

    sums = {f"{name}@{k}": 0.0 for name in ("precision", "recall", "ndcg") for k in k_list}
    n_users = 0

    for row in test_df.itertuples(index=False):
        user_id = row.user_id
        target_item = row.item_id

        seen = seen_items_per_user.get(user_id, set())
        if seen:
            mask = ~np.isin(all_items, np.fromiter(seen, dtype=all_items.dtype, count=len(seen)))
            candidate_items = all_items[mask]
        else:
            candidate_items = all_items

        scores = score_fn(user_id, candidate_items)
        rank = rank_of_item(scores, candidate_items, target_item)

        for k in k_list:
            sums[f"precision@{k}"] += precision_at_k(rank, k)
            sums[f"recall@{k}"] += recall_at_k(rank, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(rank, k)
        n_users += 1

    if n_users == 0:
        raise ValueError("test_df est vide : aucun utilisateur a evaluer.")

    return {name: total / n_users for name, total in sums.items()}


def mlflow_safe_names(metrics: dict[str, float]) -> dict[str, float]:
    """
    MLflow interdit '@' dans les noms de metriques (alphanumerique, '_', '-',
    '.', ' ', '/' uniquement). Convertit "precision@5" -> "precision_at_5"
    pour le logging MLflow uniquement -- les CSV/tableaux du rapport gardent
    la notation "@k" standard, plus lisible.
    """
    return {name.replace("@", "_at_"): value for name, value in metrics.items()}


def build_seen_items_per_user(*dfs) -> dict[int, set[int]]:
    """Fusionne les item_id vus par utilisateur sur plusieurs DataFrames (ex: train + val)."""
    seen: dict[int, set[int]] = {}
    for df in dfs:
        for user_id, item_id in zip(df["user_id"].to_numpy(), df["item_id"].to_numpy()):
            seen.setdefault(int(user_id), set()).add(int(item_id))
    return seen
