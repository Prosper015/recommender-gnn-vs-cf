"""
Split temporel strict de type Leave-One-Out (LOO).

Pour chaque utilisateur : la derniere interaction (timestamp max) va au
test, l'avant-derniere va a la validation, tout le reste va au train.
"""

import logging
from pathlib import Path

import pandas as pd

from src.data_pipeline.config import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def leave_one_out_split(
    df: pd.DataFrame,
    min_interactions: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Decoupe le DataFrame en (train, val, test) selon le protocole LOO strict.

    min_interactions : un utilisateur avec moins d'interactions que ce seuil
    va integralement dans le train (pas assez de donnees pour l'evaluer).
    """
    if min_interactions < 3:
        raise ValueError(
            "min_interactions doit etre >= 3 : il faut au moins 1 interaction "
            "pour train, 1 pour val, 1 pour test."
        )

    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Rang de chaque interaction dans l'historique de l'utilisateur (0 = plus ancienne)
    df["rank_in_user_history"] = df.groupby("user_id").cumcount()
    interaction_counts = df.groupby("user_id")["user_id"].transform("count")

    is_eligible = interaction_counts >= min_interactions
    is_test = is_eligible & (df["rank_in_user_history"] == interaction_counts - 1)
    is_val = is_eligible & (df["rank_in_user_history"] == interaction_counts - 2)
    is_train = ~(is_test | is_val)

    train_df = df.loc[is_train].drop(columns="rank_in_user_history").reset_index(drop=True)
    val_df = df.loc[is_val].drop(columns="rank_in_user_history").reset_index(drop=True)
    test_df = df.loc[is_test].drop(columns="rank_in_user_history").reset_index(drop=True)

    n_excluded_users = (~is_eligible).groupby(df["user_id"]).any().sum() if len(df) else 0

    logger.info(
        "Split LOO -> train=%d | val=%d | test=%d | utilisateurs sous le seuil (%d): %d",
        len(train_df), len(val_df), len(test_df), min_interactions, n_excluded_users,
    )

    _validate_no_leakage(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _validate_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Verifie qu'aucun utilisateur n'a, dans train, une interaction plus recente
    que sa propre interaction de test. C'est le garde-fou anti data-leakage.
    """
    if len(test_df) == 0:
        return

    max_train_ts = train_df.groupby("user_id")["timestamp"].max()
    test_ts = test_df.set_index("user_id")["timestamp"]

    common_users = max_train_ts.index.intersection(test_ts.index)
    violations = (max_train_ts.loc[common_users] > test_ts.loc[common_users]).sum()

    if violations > 0:
        raise AssertionError(
            f"Data leakage temporel detecte : {violations} utilisateur(s) ont une "
            "interaction de train posterieure a leur interaction de test."
        )
    logger.info("Verification anti-leakage OK : aucune interaction de train ne suit le test.")


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset: str,
) -> Path:
    """
    Sauvegarde les 3 splits en CSV dans data/processed/{dataset}/.

    CSV (et non parquet) volontairement : pas de dependance a pyarrow,
    portable et directement lisibles. Le backend n'a de toute facon pas
    besoin de ces splits (il consomme les modeles + movies_cleaned.csv).
    """
    out_dir = DATA_PROCESSED_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    logger.info("Splits sauvegardes dans %s", out_dir)
    return out_dir