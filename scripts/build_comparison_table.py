"""
Assemble le tableau comparatif final GNN vs baselines -- livrable
explicitement exige par le sujet ("Tableau comparatif GNN vs baselines sur
precision@k/recall@k/NDCG@k") -- a partir de :
  - results/baselines_metrics.csv  (scripts/train_baselines.py)
  - results/ablation_depth.csv     (scripts/run_ablation.py)

Usage :
    python -m scripts.build_comparison_table
"""

from __future__ import annotations

import pandas as pd

from src.data_pipeline.config import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
SELECTION_METRIC = "ndcg@10"
METRIC_COLUMNS = [
    "precision@5", "recall@5", "ndcg@5",
    "precision@10", "recall@10", "ndcg@10",
    "precision@20", "recall@20", "ndcg@20",
]


def build_comparison_table() -> pd.DataFrame:
    baselines_path = RESULTS_DIR / "baselines_metrics.csv"
    ablation_path = RESULTS_DIR / "ablation_depth.csv"

    if not baselines_path.exists():
        raise FileNotFoundError(f"{baselines_path} introuvable -- lancer scripts/train_baselines.py d'abord.")
    if not ablation_path.exists():
        raise FileNotFoundError(f"{ablation_path} introuvable -- lancer scripts/run_ablation.py d'abord.")

    baselines_df = pd.read_csv(baselines_path)

    ablation_df = pd.read_csv(ablation_path)
    best_row = ablation_df.loc[ablation_df[SELECTION_METRIC].idxmax()].copy()
    best_row["method"] = f"lightgcn (K={int(best_row['depth'])})"

    columns = ["method"] + METRIC_COLUMNS
    comparison_df = pd.concat(
        [baselines_df[columns], best_row[columns].to_frame().T],
        ignore_index=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "comparison_table.csv"
    comparison_df.to_csv(out_csv, index=False)

    out_md = RESULTS_DIR / "comparison_table.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(comparison_df.to_markdown(index=False))

    print(f"Tableau comparatif ecrit dans {out_csv} et {out_md}")
    print(comparison_df.to_string(index=False))
    return comparison_df


if __name__ == "__main__":
    build_comparison_table()
