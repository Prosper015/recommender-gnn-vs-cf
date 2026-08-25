import numpy as np
import pandas as pd

from src.models.baselines import ItemItemCFRecommender, SVDRecommender, validation_rmse

# Petit jeu synthetique : 6 utilisateurs, 5 films, assez d'interactions pour
# que scikit-surprise puisse entrainer sans erreur (KNNBasic a besoin d'au
# moins quelques co-notations par paire d'items).
TRAIN_DF = pd.DataFrame(
    {
        "user_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "item_id": [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 4, 0, 3, 4, 1, 2, 4],
        "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 5.0, 3.0, 2.0, 4.0, 3.0, 5.0, 5.0, 2.0, 4.0, 4.0, 3.0, 5.0],
    }
)
VAL_DF = pd.DataFrame(
    {"user_id": [0, 1, 2], "item_id": [3, 2, 1], "rating": [3.0, 4.0, 5.0]}
)


def test_svd_validation_rmse_is_a_positive_finite_float():
    model = SVDRecommender(n_epochs=5, random_state=42).fit(TRAIN_DF)
    rmse = validation_rmse(model, VAL_DF)
    assert isinstance(rmse, float)
    assert rmse >= 0.0
    assert np.isfinite(rmse)


def test_item_item_validation_rmse_is_a_positive_finite_float():
    model = ItemItemCFRecommender(k_neighbors=2).fit(TRAIN_DF)
    rmse = validation_rmse(model, VAL_DF)
    assert isinstance(rmse, float)
    assert rmse >= 0.0
    assert np.isfinite(rmse)


def test_svd_more_epochs_changes_the_model_predictions():
    # Pas une garantie que plus d'epoques = toujours mieux (sur un jeu aussi
    # petit ce n'est pas garanti), mais les predictions doivent au moins
    # differer entre 1 et 20 epoques -- sinon n_epochs ne servirait a rien.
    short = SVDRecommender(n_epochs=1, random_state=42).fit(TRAIN_DF)
    long = SVDRecommender(n_epochs=20, random_state=42).fit(TRAIN_DF)

    short_scores = short.score_items(0, np.array([0, 1, 2]))
    long_scores = long.score_items(0, np.array([0, 1, 2]))
    assert not np.allclose(short_scores, long_scores)
