"""
Parsing des metadonnees de films MovieLens (titres, genres) -> movies_cleaned.csv.

Complete le pipeline de donnees : contrairement a download.get_ratings()
(notes utilisateur-film), ce module lit le fichier de METADONNEES des films
(u.item pour 100k) pour produire le CSV directement attendu par le backend
(model_service.py charge data/processed/movies_cleaned.csv avec les
colonnes movieId, title, genres pour afficher les recommandations).
"""

import logging
from pathlib import Path

import pandas as pd

from src.data_pipeline.config import DATA_PROCESSED_DIR
from src.data_pipeline.download import download_zip, extract_zip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ordre exact des 19 colonnes de genre dans u.item (voir data/raw/ml-100k/README).
# "unknown" n'est pas un vrai genre (film non classe) -> exclu de la sortie.
GENRE_COLUMNS_100K = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def parse_movies_100k(extracted_dir: Path) -> pd.DataFrame:
    """
    Parse u.item (format 100k, separateur '|') -> DataFrame [movieId, title, genres].
    genres est une chaine "Genre1|Genre2|..." (vide si aucun genre marque).
    """
    item_path = extracted_dir / "u.item"
    if not item_path.exists():
        raise FileNotFoundError(f"Fichier de films introuvable : {item_path}")

    columns = ["movieId", "title", "release_date", "video_release_date", "imdb_url", *GENRE_COLUMNS_100K]
    df = pd.read_csv(
        item_path,
        sep="|",
        names=columns,
        engine="python",
        encoding="latin-1",
    )

    genre_flags = df[GENRE_COLUMNS_100K[1:]].to_numpy()  # exclut "unknown"
    real_genres = GENRE_COLUMNS_100K[1:]

    def _genres_string(row) -> str:
        matched = "|".join(g for g, flag in zip(real_genres, row) if flag == 1)
        # Chaine vide == NaN pour pandas.read_csv (cote nous ET cote backend,
        # qui relit ce CSV avec pd.read_csv standard) : sans ce fallback, un
        # film sans genre afficherait litteralement "nan" dans la demo.
        return matched or "Genre inconnu"

    df["genres"] = [_genres_string(row) for row in genre_flags]

    df["movieId"] = df["movieId"].astype("int64")
    return df[["movieId", "title", "genres"]]


def build_movies_cleaned(dataset: str = "100k") -> pd.DataFrame:
    """
    Point d'entree principal : garantit que les donnees brutes sont
    presentes (reutilise download.download_zip/extract_zip), parse les
    metadonnees, et retourne le DataFrame [movieId, title, genres].
    """
    if dataset != "100k":
        raise NotImplementedError(
            "Parsing des metadonnees implemente seulement pour 100k "
            "(format movies.dat du 1M different, non couvert ici)."
        )

    zip_path = download_zip(dataset)
    extracted_dir = extract_zip(zip_path, dataset)
    movies_df = parse_movies_100k(extracted_dir)
    logger.info("Metadonnees films chargees : %d films", len(movies_df))
    return movies_df


def save_movies_cleaned(movies_df: pd.DataFrame, out_path: Path | None = None) -> Path:
    """Sauvegarde en CSV dans data/processed/ -- chemin attendu par model_service.py (backend)."""
    out_path = out_path or (DATA_PROCESSED_DIR / "movies_cleaned.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    movies_df.to_csv(out_path, index=False)
    logger.info("movies_cleaned.csv sauvegarde : %s", out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genere movies_cleaned.csv depuis les metadonnees MovieLens.")
    parser.add_argument("--dataset", choices=["100k"], default="100k")
    args = parser.parse_args()

    df = build_movies_cleaned(args.dataset)
    save_movies_cleaned(df)
    print(df.head())
