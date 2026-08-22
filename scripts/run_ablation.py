"""
Etude d'ablation : effet de la profondeur K du GNN sur la qualite des
recommandations et sur l'over-smoothing (livrable obligatoire du sujet).

Entraine LightGCN pour chaque K in {1,2,3,4,5}, compare les metriques de
ranking (precision/recall/NDCG@k) a la similarite cosinus moyenne entre
embeddings finaux (plus elle est proche de 1, plus les embeddings sont
indiscernables -> signature de l'over-smoothing). Sauvegarde le meilleur
modele (par NDCG@10) comme models/lightgcn_best.pt.

Peut tourner en local (CPU) ou sur Colab (GPU, recommande si le CPU local
est trop lent pour 5 entrainements successifs) -- voir
notebooks/colab_train_lightgcn.ipynb.

Usage :
    python -m scripts.run_ablation --dataset 100k --epochs 50
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.data_pipeline.config import MLFLOW_TRACKING_URI, PROJECT_ROOT
from scripts.train_lightgcn import save_lightgcn, train_one_config

RESULTS_DIR = PROJECT_ROOT / "results"
DEPTHS = [1, 2, 3, 4, 5]
SELECTION_METRIC = "ndcg@10"


def run_ablation(dataset: str = "100k", epochs: int = 50, embedding_dim: int = 64) -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("model_training")

    rows = []
    best = None
    for depth in DEPTHS:
        result = train_one_config(dataset=dataset, depth=depth, epochs=epochs, embedding_dim=embedding_dim)
        rows.append({"depth": depth, **result["metrics"]})

        if best is None or result["metrics"][SELECTION_METRIC] > best["metrics"][SELECTION_METRIC]:
            best = result

    ablation_df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "ablation_depth.csv"
    ablation_df.to_csv(csv_path, index=False)
    print(f"Resultats d'ablation ecrits dans {csv_path}")

    _plot_ablation(ablation_df, RESULTS_DIR / "ablation_depth.png")

    save_lightgcn(best["model"], best["user_id_map"], best["item_id_map"])
    print(
        f"Meilleur modele (depth={best['depth']}, "
        f"{SELECTION_METRIC}={best['metrics'][SELECTION_METRIC]:.4f}) sauvegarde comme lightgcn_best.pt"
    )
    return ablation_df


def _plot_ablation(ablation_df: pd.DataFrame, out_path) -> None:
    """
    Trace cote a cote NDCG@10 (qualite) et similarite cosinus moyenne des
    embeddings (over-smoothing) en fonction de la profondeur K -- la lecture
    attendue : NDCG monte puis redescend a partir d'une certaine profondeur,
    pendant que la similarite grimpe vers 1, illustrant le compromis
    profondeur/over-smoothing exige par le sujet.
    """
    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.set_xlabel("Profondeur K (nombre de couches de convolution)")
    ax1.set_ylabel("NDCG@10", color="tab:blue")
    ax1.plot(ablation_df["depth"], ablation_df["ndcg@10"], marker="o", color="tab:blue", label="NDCG@10")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(ablation_df["depth"])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Similarite cosinus moyenne des embeddings", color="tab:red")
    ax2.plot(
        ablation_df["depth"],
        ablation_df["embedding_cosine_similarity"],
        marker="s",
        color="tab:red",
        label="Similarite cosinus (over-smoothing)",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Ablation de la profondeur LightGCN : qualite vs over-smoothing")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot d'ablation sauvegarde dans {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Etude d'ablation : profondeur LightGCN vs over-smoothing.")
    parser.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    args = parser.parse_args()
    run_ablation(dataset=args.dataset, epochs=args.epochs, embedding_dim=args.embedding_dim)
