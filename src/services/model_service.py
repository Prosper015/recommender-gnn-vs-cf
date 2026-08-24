import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

logger = logging.getLogger("recommender_api")
logging.basicConfig(level=logging.INFO)

# CHEMINS DIRECTS (racine projet = 3 niveaux au-dessus de ce fichier)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

LIGHTGCN_MODEL_PATH = MODELS_DIR / "lightgcn_best.pt"
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
ITEM_ITEM_MODEL_PATH = MODELS_DIR / "item_item_model.pkl"
MOVIES_CSV_PATH = DATA_DIR / "movies_cleaned.csv"
ID_MAPPINGS_PATH = MODELS_DIR / "id_mappings.json"


class UnknownUserError(ValueError):
    """Le user_id fourni n'existe pas dans le dataset d'entrainement (cold-start / inconnu)."""


class ModelService:
    """
    Charge les ARTEFACTS REELS (modeles entraines + mappings d'index +
    metadonnees films) et genere les recommandations cote a cote.

    Plus aucun fallback vers des donnees fictives (mock) : si un artefact est
    manquant, le modele concerne renvoie une liste vide et l'API reste
    operationnelle pour les modeles disponibles. Le healthcheck reflete
    l'etat global via `is_loaded`.
    """

    def __init__(self) -> None:
        self.lightgcn_model: Optional[Any] = None
        self.svd_model: Optional[Any] = None
        self.item_item_model: Optional[Any] = None
        self.movies_df: Optional[pd.DataFrame] = None

        # Mappings MovieLens brut <-> index internes (LightGCN)
        self.user_raw_to_internal: Dict[int, int] = {}
        self.user_internal_to_raw: Dict[int, int] = {}
        self.item_internal_to_raw: Dict[int, int] = {}

        self.is_loaded: bool = False
        self.lightgcn_available: bool = False
        self.svd_available: bool = False
        self.item_item_available: bool = False

    # ------------------------------------------------------------------ #
    # Chargement des artefacts réels
    # ------------------------------------------------------------------ #
    def load_artifacts(self) -> None:
        logger.info("Chargement des artefacts réels...")

        # 1. Métadonnées films (obligatoire pour le formatage des cartes)
        if MOVIES_CSV_PATH.exists():
            self.movies_df = pd.read_csv(MOVIES_CSV_PATH)
            logger.info("Métadonnées films chargées : %d films.", len(self.movies_df))
        else:
            logger.error("Fichier introuvable : %s (lancez `python -m scripts.run_pipeline`).", MOVIES_CSV_PATH)

        # 2. Mappings d'index (brut <-> interne) — requis pour LightGCN
        if ID_MAPPINGS_PATH.exists():
            with open(ID_MAPPINGS_PATH, "r", encoding="utf-8") as f:
                mappings = json.load(f)
            user_map = {int(k): int(v) for k, v in mappings.get("user_id_map", {}).items()}
            item_map = {int(k): int(v) for k, v in mappings.get("item_id_map", {}).items()}
            self.user_raw_to_internal = user_map
            self.user_internal_to_raw = {v: k for k, v in user_map.items()}
            self.item_internal_to_raw = {v: k for k, v in item_map.items()}
            logger.info("Mappings d'index chargés : %d users, %d items.", len(user_map), len(item_map))
        else:
            logger.error("Fichier introuvable : %s (lancez `python -m scripts.run_pipeline`).", ID_MAPPINGS_PATH)

        # 3. Modèle SVD (surprise, pickle)
        if SVD_MODEL_PATH.exists():
            with open(SVD_MODEL_PATH, "rb") as f:
                self.svd_model = pickle.load(f)
            self.svd_available = True
            logger.info("Modèle SVD chargé.")
        else:
            logger.warning("Modèle SVD absent : %s", SVD_MODEL_PATH)

        # 4. Modèle Item-Item CF (dict précalculé pickle)
        if ITEM_ITEM_MODEL_PATH.exists():
            with open(ITEM_ITEM_MODEL_PATH, "rb") as f:
                self.item_item_model = pickle.load(f)
            self.item_item_available = True
            logger.info("Modèle Item-Item chargé.")
        else:
            logger.warning("Modèle Item-Item absent : %s", ITEM_ITEM_MODEL_PATH)

        # 5. Modèle LightGCN (PyTorch)
        if LIGHTGCN_MODEL_PATH.exists():
            device = torch.device("cpu")
            self.lightgcn_model = torch.load(LIGHTGCN_MODEL_PATH, map_location=device, weights_only=False)
            self.lightgcn_model.eval()
            self.lightgcn_available = True
            logger.info("Modèle LightGCN chargé.")
        else:
            logger.warning("Modèle LightGCN absent : %s", LIGHTGCN_MODEL_PATH)

        # Prêt = au moins les métadonnées films + les mappings d'index.
        # (Les modèles peuvent être absents individuellement sans bloquer l'API.)
        self.is_loaded = (self.movies_df is not None) and bool(self.user_raw_to_internal)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def is_known_user(self, user_id: int) -> bool:
        return user_id in self.user_raw_to_internal

    def get_available_users(self, limit: Optional[int] = None) -> List[int]:
        """Liste des identifiants utilisateurs réels (MovieLens bruts) connus du système."""
        users = sorted(self.user_raw_to_internal.keys())
        if limit is not None:
            users = users[:limit]
        return users

    def _format_recommendations(self, movie_ids: List[int], scores: List[float]) -> List[Dict[str, Any]]:
        """Formate une liste d'identifiants de films (movieId BRUTS) avec leurs métadonnées."""
        if self.movies_df is None:
            return []

        recommendations = []
        for movie_id, score in zip(movie_ids, scores):
            movie_row = self.movies_df[self.movies_df["movieId"] == movie_id]
            if not movie_row.empty:
                title = movie_row.iloc[0]["title"]
                genres = movie_row.iloc[0]["genres"] if "genres" in movie_row.columns else ""
                recommendations.append({
                    "movieId": int(movie_id),
                    "title": str(title),
                    "genres": str(genres),
                    "score": round(float(score), 4),
                })
        return recommendations

    # ------------------------------------------------------------------ #
    # Recommandations par modèle (artefacts réels uniquement)
    # ------------------------------------------------------------------ #
    def recommend_lightgcn(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Recommandations via LightGCN (PyTorch), en indices internes convertis en movieId bruts."""
        if not self.lightgcn_available or self.movies_df is None:
            return []

        if not self.is_known_user(user_id):
            raise UnknownUserError(
                f"user_id={user_id} inconnu du modèle LightGCN (cold-start non géré)."
            )

        try:
            internal_user = self.user_raw_to_internal[user_id]
            with torch.no_grad():
                user_tensor = torch.tensor([internal_user], dtype=torch.long)
                if hasattr(self.lightgcn_model, "get_user_item_scores"):
                    scores = self.lightgcn_model.get_user_item_scores(user_tensor).squeeze(0)
                else:
                    scores = self.lightgcn_model(user_tensor).squeeze(0)

                top_scores, top_indices = torch.topk(scores, top_n)
                # Conversion index interne -> movieId brut avant formatage
                raw_ids = [self.item_internal_to_raw.get(int(idx), -1) for idx in top_indices.tolist()]
                return self._format_recommendations(raw_ids, top_scores.tolist())
        except UnknownUserError:
            raise
        except Exception as e:
            logger.error("Erreur lors de la recommandation LightGCN: %s", e)
            return []

    def recommend_svd(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Recommandations via SVD (Surprise) : prédit une note par film, tri décroissant."""
        if not self.svd_available or self.movies_df is None:
            return []

        try:
            all_movie_ids = self.movies_df["movieId"].unique()
            predictions = []
            for movie_id in all_movie_ids:
                if hasattr(self.svd_model, "predict"):
                    pred = self.svd_model.predict(user_id, movie_id)
                    predictions.append((movie_id, pred.est))

            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:top_n]

            movie_ids = [p[0] for p in top_predictions]
            scores = [p[1] for p in top_predictions]
            return self._format_recommendations(movie_ids, scores)
        except Exception as e:
            logger.error("Erreur lors de la recommandation SVD: %s", e)
            return []

    def recommend_item_item(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Recommandations via Item-Item CF (dict précalculé {user_id: [(item_id, score)]})."""
        if not self.item_item_available or self.movies_df is None:
            return []

        try:
            if isinstance(self.item_item_model, dict):
                if user_id not in self.item_item_model:
                    # Utilisateur non présent dans le précalcul -> aucune reco (cold-start Item-Item)
                    return []
                user_preds = self.item_item_model[user_id][:top_n]
                movie_ids = [p[0] for p in user_preds]
                scores = [p[1] for p in user_preds]
                return self._format_recommendations(movie_ids, scores)

            if hasattr(self.item_item_model, "recommend"):
                recs = self.item_item_model.recommend(user_id, top_n=top_n)
                movie_ids = [p[0] for p in recs]
                scores = [p[1] for p in recs]
                return self._format_recommendations(movie_ids, scores)

            return []
        except Exception as e:
            logger.error("Erreur lors de la recommandation Item-Item: %s", e)
            return []


# Instance globale
model_service = ModelService()
