"""
Entraine et evalue les 2 baselines classiques (SVD, Item-Item CF) sur
MovieLens. Sauvegarde les modeles dans models/ (chemin lu par le backend,
voir feature/fastapi-backend/src/services/model_service.py) et ecrit
results/baselines_metrics.csv (livrable "tableau comparatif" du sujet).

Usage :
    python -m scripts.train_baselines --dataset 100k
"""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd

from src.data_pipeline.config import MLFLOW_TRACKING_URI, PROJECT_ROOT, RANDOM_SEED
from src.data_pipeline.download import get_ratings
from src.data_pipeline.temporal_split import leave_one_out_split
from src.evaluation.metrics import build_seen_items_per_user, evaluate_full_ranking, mlflow_safe_names
from src.models.baselines import ItemItemCFRecommender, SVDRecommender, save_item_item, save_svd

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
K_LIST = [5, 10, 20]


def main(dataset: str = "100k") -> list[dict]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Experience separee de "data_pipeline" (Data Engineer) pour ne pas
    # melanger les runs des deux roles dans le meme dashboard MLflow.
    mlflow.set_experiment("model_training")

    ratings_df = get_ratings(dataset)
    train_df, val_df, test_df = leave_one_out_split(ratings_df)
    seen_items_per_user = build_seen_items_per_user(train_df, val_df)
    all_item_ids = ratings_df["item_id"].unique()  # movieId BRUTS, pas contigus

    results = []
    recommenders = {
        "svd": SVDRecommender(random_state=RANDOM_SEED),
        "item_item": ItemItemCFRecommender(),
    }

    for name, recommender in recommenders.items():
        with mlflow.start_run(run_name=f"{name}_{dataset}"):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("method", name)

            print(f"[{name}] entrainement...")
            recommender.fit(train_df)

            print(f"[{name}] evaluation full-ranking...")
            metrics = evaluate_full_ranking(
                recommender.as_score_fn(),
                test_df,
                seen_items_per_user,
                all_item_ids=all_item_ids,
                k_list=K_LIST,
            )
            mlflow.log_metrics(mlflow_safe_names(metrics))
            print(f"[{name}] {metrics}")
            results.append({"method": name, **metrics})

        if name == "svd":
            save_svd(recommender, MODELS_DIR / "svd_model.pkl")
        else:
            save_item_item(
                recommender, ratings_df["user_id"].unique(), MODELS_DIR / "item_item_model.pkl", top_n=10
            )
        print(f"[{name}] modele sauvegarde dans {MODELS_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "baselines_metrics.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Resultats ecrits dans {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraine et evalue les baselines SVD + Item-Item CF.")
    parser.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    args = parser.parse_args()
    main(args.dataset)
