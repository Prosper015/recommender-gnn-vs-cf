import sys
from pathlib import Path

# Ajoute la racine du projet au chemin de recherche des modules Python
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Importation du service singleton
from src.services.model_service import UnknownUserError, model_service

# Configuration des logs
logger = logging.getLogger("recommender_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie : charge les modèles et les métadonnées
    au démarrage de l'application FastAPI.
    """
    logger.info("Démarrage du serveur FastAPI...")
    model_service.load_artifacts()
    logger.info(
        "Artéfacts chargés. is_loaded=%s, lightgcn=%s, svd=%s, item_item=%s",
        model_service.is_loaded,
        model_service.lightgcn_available,
        model_service.svd_available,
        model_service.item_item_available,
    )
    yield
    logger.info("Arrêt du serveur FastAPI.")


# Initialisation de FastAPI
app = FastAPI(
    title="MovieLens Recommender System API",
    description="API de comparaison entre LightGCN (GNN), SVD et Item-Item CF",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuration CORS pour autoriser le Frontend React (Vite, local :5173 / Docker :3000)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health Check"])
def read_root():
    """Endpoint de santé pour vérifier le statut de l'API."""
    return {
        "status": "online",
        "message": "API de Recommandation opérationnelle",
        "models_loaded": model_service.is_loaded,
    }


@app.get("/api/v1/users", tags=["Users"])
def get_users() -> Dict[str, List[int]]:
    """Retourne la liste réelle des identifiants d'utilisateurs connus du système."""
    return {"users": model_service.get_available_users()}


@app.get("/api/v1/recommendations/{user_id}", tags=["Recommendations"])
def get_recommendations(
    user_id: int,
    top_n: int = Query(default=5, ge=1, le=20, description="Nombre de recommandations par modèle")
) -> Dict[str, Any]:
    """
    Génère les recommandations de films pour un utilisateur donné 
    en utilisant simultanément LightGCN, SVD et Item-Item CF.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Les artefacts de recommandation ne sont pas chargés. "
                   "Lancez `python -m scripts.run_pipeline` puis redémarrez l'API."
        )

    # Cold-start / utilisateur inconnu -> 404 explicite
    if not model_service.is_known_user(user_id):
        raise HTTPException(
            status_code=404,
            detail=f"user_id={user_id} est inconnu du dataset d'entraînement (cold-start non géré)."
        )

    try:
        lightgcn_recs = model_service.recommend_lightgcn(user_id=user_id, top_n=top_n)
        svd_recs = model_service.recommend_svd(user_id=user_id, top_n=top_n)
        item_item_recs = model_service.recommend_item_item(user_id=user_id, top_n=top_n)

        return {
            "user_id": user_id,
            "top_n": top_n,
            "lightgcn": lightgcn_recs,
            "svd": svd_recs,
            "item_item": item_item_recs,
        }
    except Exception as e:
        logger.error(f"Erreur lors de la génération des recommandations pour user {user_id} : {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du serveur lors du calcul des recommandations : {str(e)}"
        )