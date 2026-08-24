"""
Pipeline complet de (de)mockage : genere TOUS les artefacts reels consommes
par le backend FastAPI a partir de MovieLens.

En une seule commande :
    1. Telecharge + parse ratings et films (MovieLens)
    2. Split temporel Leave-One-Out + sauvegarde des splits
    3. Entraine et exporte les 3 modeles (SVD, Item-Item CF, LightGCN)
    4. Exporte movies_cleaned.csv + id_mappings.json

C'est le script a lancer une fois (en local ou dans le Dockerfile) pour
remplacer les donnees fictives par de vraies predictions.

Usage :
    python -m scripts.run_pipeline --dataset 100k
"""

from __future__ import annotations

import argparse

from src.data_pipeline.config import DATA_PROCESSED_DIR, PROJECT_ROOT
from src.data_pipeline.download import get_movies, get_ratings
from src.data_pipeline.temporal_split import leave_one_out_split, save_splits

from scripts.train_baselines import main as train_baselines
from scripts.train_lightgcn import save_lightgcn, train_one_config


def run_pipeline(dataset: str = "100k", depth: int = 3, epochs: int = 50, use_mlflow: bool = True) -> None:
    print(f"[pipeline] Dataset={dataset} | generation des artefacts REELS...")

    # 1. Donnees brutes (ratings + films)
    ratings_df = get_ratings(dataset)
    get_movies(dataset)

    # 2. Split temporel LOO (anti-leakage)
    train_df, val_df, test_df = leave_one_out_split(ratings_df)
    save_splits(train_df, val_df, test_df, dataset)

    # 3. Baselines SVD + Item-Item CF (working sur movieId BRUTS)
    train_baselines(dataset, use_mlflow=use_mlflow)

    # 4. LightGCN (reindexe + graphe bipartite + BPR)
    result = train_one_config(
        dataset=dataset,
        depth=depth,
        epochs=epochs,
        log_to_mlflow=use_mlflow,
    )
    save_lightgcn(result["model"], result["user_id_map"], result["item_id_map"])

    print("[pipeline] Terminé. Artefacts dans :")
    print(f"   - modeles : {PROJECT_ROOT / 'models'}")
    print(f"   - donnees  : {DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genere tous les artefacts reels du projet.")
    parser.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    parser.add_argument("--depth", type=int, default=3, help="Profondeur K du LightGCN")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--no-mlflow", action="store_true", help="Desactive le logging MLflow")
    args = parser.parse_args()

    run_pipeline(
        dataset=args.dataset,
        depth=args.depth,
        epochs=args.epochs,
        use_mlflow=not args.no_mlflow,
    )
