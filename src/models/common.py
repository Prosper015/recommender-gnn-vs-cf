"""
Interface commune aux 3 methodes de recommandation (SVD, Item-Item CF, LightGCN).

Avoir une seule interface permet de :
  - brancher indifferemment n'importe laquelle des 3 methodes dans
    evaluate_full_ranking() (src/evaluation/metrics.py) ;
  - generer le tableau comparatif (scripts/build_comparison_table.py) sans
    code specifique par methode ;
  - satisfaire directement le contrat attendu cote backend par
    model_service.py (methode .recommend(user_id, top_n=...)).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


class Recommender(ABC):
    """
    Contrat : apres fit(), l'implementation doit avoir peuple
    `self.all_item_ids_` (np.ndarray des item_id vus en train, univers de
    candidats pour recommend()).
    """

    all_item_ids_: np.ndarray

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "Recommender":
        """Entraine le modele sur le DataFrame train (colonnes user_id, item_id, rating, ...)."""

    @abstractmethod
    def score_items(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        """
        Score un ensemble d'items pour un utilisateur donne.
        Retourne un tableau de meme longueur que item_ids ; plus haut = plus pertinent.
        Utilise comme `score_fn` par evaluate_full_ranking().
        """

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        exclude_items: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Top-N items recommandes pour user_id, tries par score decroissant.
        Implementation par defaut : score tout l'univers d'items connu moins
        exclude_items. Les sous-classes peuvent la surcharger si elles ont un
        chemin plus efficace.
        """
        candidates = self.all_item_ids_
        if exclude_items:
            exclude_set = set(exclude_items)
            candidates = np.array([i for i in candidates if i not in exclude_set])

        if candidates.size == 0:
            return []

        scores = self.score_items(user_id, candidates)
        order = np.argsort(-scores, kind="stable")[:top_n]
        return [(int(candidates[i]), float(scores[i])) for i in order]

    def as_score_fn(self) -> Callable[[int, np.ndarray], np.ndarray]:
        """Adaptateur direct pour evaluate_full_ranking(score_fn=...)."""
        return self.score_items
