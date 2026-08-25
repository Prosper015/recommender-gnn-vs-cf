import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import torch

logger = logging.getLogger("recommender_api")
logging.basicConfig(level=logging.INFO)

# CHEMINS DIRECTS
BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

LIGHTGCN_MODEL_PATH = MODELS_DIR / "lightgcn_best.pt"
SVD_MODEL_PATH = MODELS_DIR / "svd_model.pkl"
ITEM_ITEM_MODEL_PATH = MODELS_DIR / "item_item_model.pkl"
MOVIES_CSV_PATH = DATA_DIR / "movies_cleaned.csv"


class ModelService:
    def __init__(self) -> None:
        self.lightgcn_model: Optional[Any] = None
        self.svd_model: Optional[Any] = None
        self.item_item_model: Optional[Any] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.is_loaded: bool = False

    def load_artifacts(self) -> None:
        logger.info("Chargement des modèles...")

        # 1. Chargement des films
        if MOVIES_CSV_PATH.exists():
            self.movies_df = pd.read_csv(MOVIES_CSV_PATH)
            logger.info(" Métadonnées des films chargées.")

        # 2. Chargement SVD
        if SVD_MODEL_PATH.exists():
            with open(SVD_MODEL_PATH, "rb") as f:
                self.svd_model = pickle.load(f)
            logger.info(" Modèle SVD chargé.")

        # 3. Chargement Item-Item CF
        if ITEM_ITEM_MODEL_PATH.exists():
            with open(ITEM_ITEM_MODEL_PATH, "rb") as f:
                self.item_item_model = pickle.load(f)
            logger.info(" Modèle Item-Item chargé.")

        # 4. Chargement LightGCN (PyTorch)
        if LIGHTGCN_MODEL_PATH.exists():
            device = torch.device("cpu")
            self.lightgcn_model = torch.load(LIGHTGCN_MODEL_PATH, map_location=device, weights_only=False)
            logger.info(" Modèle LightGCN chargé.")

        self.is_loaded = True

    def _format_recommendations(self, movie_ids: List[int], scores: List[float]) -> List[Dict[str, Any]]:
        """Formate la liste d'identifiants de films avec leurs métadonnées."""
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
                    "score": round(float(score), 4)
                })
        return recommendations

    def _generate_mock_recommendations(self, user_id: int, top_n: int = 5, model_name: str = "") -> List[Dict[str, Any]]:
        mock_movies = [
            {"movieId": 1, "title": "Inception", "genres": "Sci-Fi|Thriller"},
            {"movieId": 2, "title": "The Dark Knight", "genres": "Action|Crime|Drama"},
            {"movieId": 3, "title": "Interstellar", "genres": "Sci-Fi|Adventure|Drama"},
            {"movieId": 4, "title": "Pulp Fiction", "genres": "Crime|Drama"},
            {"movieId": 5, "title": "The Matrix", "genres": "Sci-Fi|Action"},
            {"movieId": 6, "title": "Forrest Gump", "genres": "Drama|Romance"},
            {"movieId": 7, "title": "Fight Club", "genres": "Drama|Thriller"},
            {"movieId": 8, "title": "Gladiator", "genres": "Action|Adventure|Drama"},
            {"movieId": 9, "title": "The Shawshank Redemption", "genres": "Drama"},
            {"movieId": 10, "title": "Titanic", "genres": "Drama|Romance"},
        ]

        base_scores = {"lightgcn": 0.95, "svd": 4.2, "item_item": 0.88}
        base = base_scores.get(model_name, 0.9)

        recommendations = []
        for i in range(min(top_n, len(mock_movies))):
            movie = mock_movies[(user_id + i) % len(mock_movies)]
            score = round(base - (i * 0.05) + ((user_id + i) % 3) * 0.01, 4)
            recommendations.append({
                "movieId": movie["movieId"],
                "title": movie["title"],
                "genres": movie["genres"],
                "score": score,
            })
        return recommendations

    def recommend_lightgcn(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Génère les recommandations via LightGCN (PyTorch)."""
        if self.lightgcn_model is None:
            return self._generate_mock_recommendations(user_id, top_n, "lightgcn")

        try:
            with torch.no_grad():
                user_tensor = torch.tensor([user_id], dtype=torch.long)
                if hasattr(self.lightgcn_model, "get_user_item_scores"):
                    scores = self.lightgcn_model.get_user_item_scores(user_tensor).squeeze(0)
                else:
                    scores = self.lightgcn_model(user_tensor).squeeze(0)

                top_scores, top_indices = torch.topk(scores, top_n)
                return self._format_recommendations(top_indices.tolist(), top_scores.tolist())
        except Exception as e:
            logger.error(f"Erreur lors de la recommandation LightGCN: {e}")
            return self._generate_mock_recommendations(user_id, top_n, "lightgcn")

    def recommend_svd(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Génère les recommandations via SVD (Surprise / Scikit-Learn)."""
        if self.svd_model is None or self.movies_df is None:
            return self._generate_mock_recommendations(user_id, top_n, "svd")

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
            logger.error(f"Erreur lors de la recommandation SVD: {e}")
            return self._generate_mock_recommendations(user_id, top_n, "svd")

    def recommend_item_item(self, user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Génère les recommandations via Item-Item Collaborative Filtering."""
        if self.item_item_model is None or self.movies_df is None:
            return self._generate_mock_recommendations(user_id, top_n, "item_item")

        try:
            if hasattr(self.item_item_model, "recommend"):
                return self.item_item_model.recommend(user_id, top_n=top_n)

            if isinstance(self.item_item_model, dict) and user_id in self.item_item_model:
                user_preds = self.item_item_model[user_id][:top_n]
                movie_ids = [p[0] for p in user_preds]
                scores = [p[1] for p in user_preds]
                return self._format_recommendations(movie_ids, scores)

            return []
        except Exception as e:
            logger.error(f"Erreur lors de la recommandation Item-Item: {e}")
            return self._generate_mock_recommendations(user_id, top_n, "item_item")


# Instance globale
model_service = ModelService()