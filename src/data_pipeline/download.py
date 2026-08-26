"""
Telechargement et parsing des jeux de donnees MovieLens (100K et 1M).

Usage en ligne de commande :
    python -m src.data_pipeline.download --dataset 100k
    python -m src.data_pipeline.download --dataset 1m

Usage en import :
    from src.data_pipeline.download import get_ratings
    df = get_ratings("100k")   # telecharge si besoin, puis parse
"""

import argparse
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.data_pipeline.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MOVIELENS_EXTRACTED_DIRNAME,
    MOVIELENS_URLS,
    STANDARD_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ordre canonique des 19 genres de MovieLens 100k (colonnes 6..24 de u.item).
GENRES_100K = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _validate_dataset_name(dataset: str) -> str:
    dataset = dataset.lower().strip()
    if dataset not in MOVIELENS_URLS:
        raise ValueError(
            f"Dataset '{dataset}' inconnu. Valeurs acceptees : {list(MOVIELENS_URLS.keys())}"
        )
    return dataset


def download_zip(dataset: str, force: bool = False) -> Path:
    """Telecharge l'archive zip de MovieLens si elle n'existe pas deja localement."""
    dataset = _validate_dataset_name(dataset)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    url = MOVIELENS_URLS[dataset]
    zip_path = DATA_RAW_DIR / f"ml-{dataset}.zip"

    if zip_path.exists() and not force:
        logger.info("Archive deja presente : %s (utiliser force=True pour retelecharger)", zip_path)
        return zip_path

    logger.info("Telechargement de %s ...", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=zip_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Telechargement termine : %s", zip_path)
    return zip_path


def extract_zip(zip_path: Path, dataset: str, force: bool = False) -> Path:
    """Dezippe l'archive et retourne le dossier extrait."""
    dataset = _validate_dataset_name(dataset)
    extracted_dir = DATA_RAW_DIR / MOVIELENS_EXTRACTED_DIRNAME[dataset]

    if extracted_dir.exists() and not force:
        logger.info("Dossier deja extrait : %s", extracted_dir)
        return extracted_dir

    logger.info("Extraction de %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_RAW_DIR)

    return extracted_dir


def parse_ratings(extracted_dir: Path, dataset: str) -> pd.DataFrame:
    """
    Parse le fichier de notes brut vers un DataFrame standardise :
    colonnes = [user_id, item_id, rating, timestamp]

    Les deux versions de MovieLens ont des formats de fichiers differents :
    - 100k : fichier 'u.data', separateur TAB
    - 1m   : fichier 'ratings.dat', separateur '::'
    """
    dataset = _validate_dataset_name(dataset)

    if dataset == "100k":
        ratings_path = extracted_dir / "u.data"
        df = pd.read_csv(
            ratings_path,
            sep="\t",
            names=STANDARD_COLUMNS,
            engine="python",
        )
    elif dataset == "1m":
        ratings_path = extracted_dir / "ratings.dat"
        df = pd.read_csv(
            ratings_path,
            sep="::",
            names=STANDARD_COLUMNS,
            engine="python",
            encoding="latin-1",
        )

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Fichier de notes introuvable : {ratings_path}. "
            "Verifiez que l'extraction s'est bien passee."
        )

    # Types explicites : evite les surprises silencieuses en aval (split, graphe)
    df["user_id"] = df["user_id"].astype("int64")
    df["item_id"] = df["item_id"].astype("int64")
    df["rating"] = df["rating"].astype("float32")
    df["timestamp"] = df["timestamp"].astype("int64")

    _sanity_check(df)
    return df


def _sanity_check(df: pd.DataFrame) -> None:
    """Verifications minimales de qualite avant de considerer les donnees propres."""
    assert not df.isna().any().any(), "Valeurs manquantes detectees dans les ratings."
    assert df["rating"].between(0.5, 5.0).all(), "Ratings hors de l'echelle attendue [0.5, 5]."
    n_duplicates = df.duplicated(subset=["user_id", "item_id", "timestamp"]).sum()
    if n_duplicates > 0:
        logger.warning("%d lignes dupliquees (user_id, item_id, timestamp) detectees.", n_duplicates)


def reindex_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Reindexe user_id et item_id en entiers contigus [0, N-1].

    Indispensable pour la construction du graphe (PyG/DGL attendent des
    indices de noeuds contigus a partir de 0) et pour les embeddings du
    ML Engineer (nn.Embedding(num_users, dim) exige des indices bornes).

    Retourne le DataFrame reindexe + les deux mappings (utiles pour retrouver
    l'ID original MovieLens depuis l'indice interne, ex: cote API/Frontend).
    """
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_id_map = {raw_id: idx for idx, raw_id in enumerate(unique_users)}
    item_id_map = {raw_id: idx for idx, raw_id in enumerate(unique_items)}

    df = df.copy()
    df["user_id"] = df["user_id"].map(user_id_map)
    df["item_id"] = df["item_id"].map(item_id_map)

    return df, user_id_map, item_id_map


def get_ratings(dataset: str = "100k", force: bool = False) -> pd.DataFrame:
    """
    Point d'entree principal : garantit que les donnees brutes sont
    presentes localement (telecharge sinon), puis retourne le DataFrame parse.

    Ne fait PAS le reindexing ni le split : cette fonction retourne les
    ratings "propres" avec les IDs MovieLens originaux. Le reindexing est
    fait explicitement dans le pipeline (voir pipeline.py) pour que chaque
    etape reste testable independamment.
    """
    dataset = _validate_dataset_name(dataset)
    zip_path = download_zip(dataset, force=force)
    extracted_dir = extract_zip(zip_path, dataset, force=force)
    df = parse_ratings(extracted_dir, dataset)
    logger.info(
        "Dataset %s charge : %d interactions, %d utilisateurs, %d items",
        dataset, len(df), df["user_id"].nunique(), df["item_id"].nunique(),
    )
    return df


def parse_movies(extracted_dir: Path, dataset: str) -> pd.DataFrame:
    """
    Parse les metadonnees films vers un DataFrame standardise :
    colonnes = [movieId, title, genres] (genres separes par '|').

    - 100k : fichier 'u.item', genres en 19 colonnes binaires.
    - 1m   : fichier 'movies.dat', genre deja en texte '|'-sep.
    """
    dataset = _validate_dataset_name(dataset)

    if dataset == "100k":
        movies_path = extracted_dir / "u.item"
        cols = ["movieId", "title", "release_date", "video_release_date", "imdb_url"] + GENRES_100K
        movies = pd.read_csv(
            movies_path, sep="|", names=cols, encoding="latin-1", engine="python",
            usecols=range(len(cols)),
        )
        genre_flags = movies[GENRES_100K].fillna(0).astype(int)
        movies["genres"] = genre_flags.apply(
            lambda row: "|".join([g for g in GENRES_100K if int(row[g]) == 1 and g != "unknown"]),
            axis=1,
        ).replace("", "Genre inconnu")
        movies = movies[["movieId", "title", "genres"]]
    elif dataset == "1m":
        movies_path = extracted_dir / "movies.dat"
        movies = pd.read_csv(
            movies_path, sep="::", names=["movieId", "title", "genres"],
            encoding="latin-1", engine="python",
        )

    if not movies_path.exists():
        raise FileNotFoundError(f"Fichier de films introuvable : {movies_path}.")

    movies["movieId"] = movies["movieId"].astype("int64")
    movies["title"] = movies["title"].astype(str).str.strip()
    movies["genres"] = movies["genres"].astype(str)
    return movies.reset_index(drop=True)


def save_movies(movies: pd.DataFrame, out_path: Path) -> None:
    """Sauvegarde le Catalogue de films consomme par le backend (movies_cleaned.csv)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(out_path, index=False)
    logger.info("Metadonnees films sauvegardees : %s (%d films)", out_path, len(movies))


def get_movies(dataset: str = "100k", force: bool = False) -> pd.DataFrame:
    """
    Point d'entree : garantit que le dataset brut est present, parse les films
    et les sauvegarde dans data/processed/movies_cleaned.csv. Retourne le DF.
    """
    dataset = _validate_dataset_name(dataset)
    zip_path = download_zip(dataset, force=force)
    extracted_dir = extract_zip(zip_path, dataset, force=force)
    movies = parse_movies(extracted_dir, dataset)
    save_movies(movies, DATA_PROCESSED_DIR / "movies_cleaned.csv")
    return movies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecharge et parse MovieLens.")
    parser.add_argument("--dataset", choices=["100k", "1m"], default="100k")
    parser.add_argument("--force", action="store_true", help="Force le re-telechargement")
    args = parser.parse_args()

    ratings_df = get_ratings(args.dataset, force=args.force)
    print(ratings_df.head())
    print(ratings_df.describe())