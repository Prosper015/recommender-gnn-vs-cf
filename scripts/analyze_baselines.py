"""
Analyse des baselines : deux preuves distinctes, a ne pas confondre.

1. Courbes de diagnostic par methode (RMSE sur validation) :
   - SVD : RMSE vs n_epochs -- montre la convergence de l'entrainement
     (justifie le choix de n_epochs=20).
   - Item-Item CF : RMSE vs k_neighbors -- montre la sensibilite au nombre
     de voisins (justifie le choix de k=40). Pas une "courbe d'entrainement"
     (cet algorithme n'a pas d'epoques), mais un diagnostic equivalent :
     comment le resultat varie avec l'hyperparametre.

2. Graphique de comparaison SVD vs Item-Item CF, sur les MEMES metriques de
   ranking (precision/recall/NDCG@k), sur le MEME jeu de test cache -- c'est
   la vraie preuve de comparaison exigee par le sujet, a partir des
   resultats deja produits par scripts/train_baselines.py.

Usage :
    python -m scripts.train_baselines --dataset 100k    (a lancer d'abord)
    python -m scripts.analyze_baselines --dataset 100k
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from src.data_pipeline.config import PROJECT_ROOT
from src.data_pipeline.download import get_ratings
from src.data_pipeline.temporal_split import leave_one_out_split
from src.models.baselines import ItemItemCFRecommender, SVDRecommender, validation_rmse

RESULTS_DIR = PROJECT_ROOT / "results"

SVD_EPOCH_VALUES = [1, 5, 10, 15, 20, 30]
ITEM_ITEM_K_VALUES = [5, 10, 20, 40, 80]

METRIC_COLUMNS = [
    "precision@5", "recall@5", "ndcg@5",
    "precision@10", "recall@10", "ndcg@10",
    "precision@20", "recall@20", "ndcg@20",
]


def svd_convergence_curve(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_epochs in SVD_EPOCH_VALUES:
        model = SVDRecommender(n_epochs=n_epochs).fit(train_df)
        rmse = validation_rmse(model, val_df)
        print(f"[svd] n_epochs={n_epochs} -> RMSE validation = {rmse:.4f}")
        rows.append({"n_epochs": n_epochs, "rmse": rmse})
    return pd.DataFrame(rows)


def item_item_sensitivity_curve(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in ITEM_ITEM_K_VALUES:
        model = ItemItemCFRecommender(k_neighbors=k).fit(train_df)
        rmse = validation_rmse(model, val_df)
        print(f"[item_item] k_neighbors={k} -> RMSE validation = {rmse:.4f}")
        rows.append({"k_neighbors": k, "rmse": rmse})
    return pd.DataFrame(rows)


def plot_hyperparameter_curves(svd_df: pd.DataFrame, item_item_df: pd.DataFrame, out_path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(svd_df["n_epochs"], svd_df["rmse"], marker="o", color="tab:blue")
    ax1.set_xlabel("Nombre d'epoques (n_epochs)")
    ax1.set_ylabel("RMSE (validation)")
    ax1.set_title("SVD : convergence de l'entrainement")
    ax1.set_xticks(svd_df["n_epochs"])

    ax2.plot(item_item_df["k_neighbors"], item_item_df["rmse"], marker="s", color="tab:orange")
    ax2.set_xlabel("Nombre de voisins (k)")
    ax2.set_ylabel("RMSE (validation)")
    ax2.set_title("Item-Item CF : sensibilite a k")
    ax2.set_xticks(item_item_df["k_neighbors"])

    fig.suptitle("Baselines : diagnostics d'hyperparametres (RMSE sur validation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Courbes de diagnostic sauvegardees dans {out_path}")


def plot_comparison_bars(baselines_metrics_path, out_path) -> None:
    df = pd.read_csv(baselines_metrics_path).set_index("method")

    x = range(len(METRIC_COLUMNS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], df.loc["svd", METRIC_COLUMNS], width, label="SVD", color="tab:blue")
    ax.bar(
        [i + width / 2 for i in x], df.loc["item_item", METRIC_COLUMNS], width,
        label="Item-Item CF", color="tab:orange",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(METRIC_COLUMNS, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Comparaison SVD vs Item-Item CF (protocole full-ranking, test cache)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Graphique de comparaison sauvegarde dans {out_path}")


def main(dataset: str = "100k") -> None:
    baselines_metrics_path = RESULTS_DIR / "baselines_metrics.csv"
    if not baselines_metrics_path.exists():
        raise FileNotFoundError(
            f"{baselines_metrics_path} introuvable -- lancer scripts/train_baselines.py d'abord."
        )

    ratings_df = get_ratings(dataset)
    train_df, val_df, _test_df = leave_one_out_split(ratings_df)

    print("--- Courbe de convergence SVD ---")
    svd_df = svd_convergence_curve(train_df, val_df)
    print("--- Courbe de sensibilite Item-Item CF ---")
    item_item_df = item_item_sensitivity_curve(train_df, val_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    svd_df.to_csv(RESULTS_DIR / "svd_convergence.csv", index=False)
    item_item_df.to_csv(RESULTS_DIR / "item_item_sensitivity.csv", index=False)

    plot_hyperparameter_curves(svd_df, item_item_df, RESULTS_DIR / "baselines_hyperparameter_curves.png")
    plot_comparison_bars(baselines_metrics_path, RESULTS_DIR / "baselines_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Courbes de diagnostic + comparaison des baselines.")
    parser.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    args = parser.parse_args()
    main(args.dataset)
