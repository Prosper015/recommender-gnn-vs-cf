"""
Entraine UNE configuration LightGCN (profondeur K fixee), evalue sur le
protocole full-ranking, et sauvegarde le modele + le mapping d'IDs dans models/.

Fonctionne identiquement en local (CPU) et sur Colab (GPU) : le device est
detecte automatiquement (torch.cuda.is_available()), aucune option
specifique a ajouter d'un environnement a l'autre.

Usage :
    python -m scripts.train_lightgcn --dataset 100k --depth 3 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import random

# MLflow est optionnel : le suivi d'experiences n'est pas requis pour
# produire les artefacts de production. Si MLflow n'est pas installe,
# l'entrainement se deroule normalement sans logging.
try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None

import numpy as np
import torch

from src.data_pipeline.config import MLFLOW_TRACKING_URI, PROJECT_ROOT, RANDOM_SEED
from src.data_pipeline.download import get_ratings, reindex_ids
from src.data_pipeline.graph_builder import build_bipartite_graph
from src.data_pipeline.temporal_split import leave_one_out_split
from src.evaluation.metrics import build_seen_items_per_user, evaluate_full_ranking, mlflow_safe_names
from src.models.lightgcn import LightGCN, LightGCNRecommender, sample_bpr_triplets

MODELS_DIR = PROJECT_ROOT / "models"
K_LIST = [5, 10, 20]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_reindexed_data(dataset: str):
    """
    Charge MovieLens et reindexe user_id/item_id en entiers contigus (requis
    pour les embeddings + le graphe), puis applique le meme protocole LOO
    temporel que les baselines. Le split se fait par rang temporel par
    utilisateur : reindexer avant ou apres ne change pas quelles
    interactions tombent en train/val/test, seulement leurs IDs.
    """
    ratings_df = get_ratings(dataset)
    reindexed_df, user_id_map, item_id_map = reindex_ids(ratings_df)
    train_df, val_df, test_df = leave_one_out_split(reindexed_df)
    return train_df, val_df, test_df, user_id_map, item_id_map


def train_one_config(
    dataset: str,
    depth: int,
    epochs: int,
    embedding_dim: int = 64,
    lr: float = 1e-3,
    l2_reg: float = 1e-4,
    seed: int = RANDOM_SEED,
    log_to_mlflow: bool = True,
) -> dict:
    """
    Entraine une configuration LightGCN et retourne un dict {model, metrics,
    user_id_map, item_id_map, depth}. Fonction importable telle quelle par
    scripts/run_ablation.py (une config = un appel).
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[depth={depth}] device={device}")

    train_df, val_df, test_df, user_id_map, item_id_map = load_reindexed_data(dataset)
    num_users, num_items = len(user_id_map), len(item_id_map)

    graph = build_bipartite_graph(train_df, num_users=num_users, num_items=num_items)

    model = LightGCN(num_users, num_items, embedding_dim=embedding_dim, num_layers=depth).to(device)
    model.set_graph(graph.edge_index)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    seen_items_per_user = build_seen_items_per_user(train_df, val_df)

    run_ctx = (
        mlflow.start_run(run_name=f"lightgcn_k{depth}_{dataset}")
        if (mlflow is not None and log_to_mlflow)
        else None
    )
    if run_ctx is not None:
        run_ctx.__enter__()
        mlflow.log_params(
            {
                "dataset": dataset,
                "method": "lightgcn",
                "depth": depth,
                "epochs": epochs,
                "embedding_dim": embedding_dim,
                "lr": lr,
                "l2_reg": l2_reg,
                "seed": seed,
            }
        )

    try:
        model.train()
        log_every = max(1, epochs // 10)
        for epoch in range(1, epochs + 1):
            users, pos_items, neg_items = sample_bpr_triplets(train_df, num_items, rng)
            optimizer.zero_grad()
            loss = model.bpr_loss(
                torch.from_numpy(users).long().to(device),
                torch.from_numpy(pos_items).long().to(device),
                torch.from_numpy(neg_items).long().to(device),
                l2_reg=l2_reg,
            )
            loss.backward()
            optimizer.step()

            if epoch % log_every == 0 or epoch == epochs:
                print(f"[depth={depth}] epoch {epoch}/{epochs} - loss={loss.item():.4f}")
                if run_ctx is not None:
                    mlflow.log_metric("train_loss", loss.item(), step=epoch)

        model.eval()
        recommender = LightGCNRecommender(model)
        metrics = evaluate_full_ranking(
            recommender.as_score_fn(),
            test_df,
            seen_items_per_user,
            all_item_ids=np.arange(num_items),
            k_list=K_LIST,
        )
        metrics["embedding_cosine_similarity"] = model.embedding_similarity_diagnostic()
        print(f"[depth={depth}] {metrics}")

        if run_ctx is not None:
            mlflow.log_metrics(mlflow_safe_names(metrics))
    finally:
        if run_ctx is not None:
            run_ctx.__exit__(None, None, None)

    return {
        "model": model,
        "metrics": metrics,
        "user_id_map": user_id_map,
        "item_id_map": item_id_map,
        "depth": depth,
    }


def save_lightgcn(model: LightGCN, user_id_map: dict, item_id_map: dict, out_dir=MODELS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Objet MODELE COMPLET (pas seulement le state_dict) : model_service.py
    # (feature/fastapi-backend) fait torch.load(path) et appelle directement
    # .get_user_item_scores() sur l'objet recharge.
    torch.save(model.to("cpu"), out_dir / "lightgcn_best.pt")

    # Comble le trou d'integration identifie entre branches : le mapping
    # movieId brut <-> index interne n'etait exporte nulle part, alors que
    # get_user_item_scores() retourne des scores en indices internes.
    id_mappings = {
        "user_id_map": {str(k): v for k, v in user_id_map.items()},
        "item_id_map": {str(k): v for k, v in item_id_map.items()},
    }
    with open(out_dir / "id_mappings.json", "w", encoding="utf-8") as f:
        json.dump(id_mappings, f, ensure_ascii=False, indent=2)
    print(f"Modele + id_mappings.json sauvegardes dans {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraine une configuration LightGCN.")
    parser.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    parser.add_argument("--depth", type=int, default=3, help="Nombre de couches de convolution K")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-reg", type=float, default=1e-4)
    parser.add_argument("--no-mlflow", action="store_true", help="Desactive le logging MLflow")
    args = parser.parse_args()

    if mlflow is not None and not args.no_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("model_training")

    result = train_one_config(
        dataset=args.dataset,
        depth=args.depth,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        l2_reg=args.l2_reg,
        log_to_mlflow=(mlflow is not None and not args.no_mlflow),
    )
    save_lightgcn(result["model"], result["user_id_map"], result["item_id_map"])
