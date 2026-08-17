"""
Integration MLflow pour le pipeline de donnees.

Trace les parametres, statistiques et fichiers produits a chaque execution
du pipeline, pour garder une tracabilite complete (exigence du professeur).
"""

import logging

import mlflow

from src.data_pipeline.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def init_mlflow() -> None:
    """A appeler une fois en debut de script, avant tout run."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow configure -> tracking_uri=%s | experiment=%s",
                MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME)


def log_pipeline_run(
    dataset: str,
    min_interactions: int,
    ratings_df,
    train_df,
    val_df,
    test_df,
    graph_summary: dict,
    artifact_paths: list,
) -> str:
    """
    Log un run MLflow complet pour une execution du pipeline de donnees.

    Retourne le run_id, utile pour le chainer avec le run d'entrainement
    du ML Engineer plus tard.
    """
    with mlflow.start_run(run_name=f"data_pipeline_{dataset}") as run:
        # Parametres : ce qui definit la version des donnees
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("min_interactions", min_interactions)
        mlflow.log_param("split_strategy", "leave_one_out_temporal")

        # Metriques descriptives
        mlflow.log_metric("n_interactions_total", len(ratings_df))
        mlflow.log_metric("n_users", ratings_df["user_id"].nunique())
        mlflow.log_metric("n_items", ratings_df["item_id"].nunique())
        mlflow.log_metric("n_train", len(train_df))
        mlflow.log_metric("n_val", len(val_df))
        mlflow.log_metric("n_test", len(test_df))

        density = len(ratings_df) / (ratings_df["user_id"].nunique() * ratings_df["item_id"].nunique())
        mlflow.log_metric("matrix_density", density)

        for key, value in graph_summary.items():
            mlflow.log_metric(f"graph_{key}", value)

        # Artefacts : fichiers reutilisables par le ML Engineer
        for path in artifact_paths:
            mlflow.log_artifact(str(path))

        logger.info("Run MLflow logue : run_id=%s", run.info.run_id)
        return run.info.run_id