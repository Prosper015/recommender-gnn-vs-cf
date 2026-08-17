"""
Configuration centrale du pipeline de données.

Toutes les fonctions du pipeline (download, split, graph) lisent leurs
chemins ici. Un seul endroit à modifier si l'arborescence change.
"""

from pathlib import Path

# Racine du projet = 3 niveaux au-dessus de ce fichier
# (src/data_pipeline/config.py -> src/data_pipeline -> src -> racine)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# URLs officielles GroupLens
MOVIELENS_URLS = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}

# Nom du dossier une fois dézippé (change de format entre 100k et 1m)
MOVIELENS_EXTRACTED_DIRNAME = {
    "100k": "ml-100k",
    "1m": "ml-1m",
}

# Où MLflow écrit ses runs. Backend SQLite local (recommandé par MLflow
# depuis 2.x, le backend fichier "mlruns/" pur est en maintenance limitée) ;
# peut être remplacé par une URI serveur (http://...) en production.
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "data_pipeline"

# Colonnes standardisées utilisées dans TOUT le projet, quel que soit
# le dataset source (100k ou 1m). Le ML Engineer et le Dev peuvent donc
# compter sur ce schéma fixe en sortie du data_pipeline.
STANDARD_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]

RANDOM_SEED = 42

DEFAULT_DATASET = "100k"