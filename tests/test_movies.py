from pathlib import Path

from src.data_pipeline.download import parse_movies

U_ITEM_SAMPLE = (
    "1|Toy Story (1995)|01-Jan-1995||http://example.com/toy-story"
    "|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
    "2|Se7en (1995)|22-Sep-1995||http://example.com/se7en"
    "|0|0|0|0|0|0|1|0|1|0|0|0|0|0|0|0|1|0|0\n"
    "3|Mystery Movie|01-Jan-1996||http://example.com/mystery"
    "|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
)


def test_parse_movies_100k_extracts_title_and_genres(tmp_path: Path):
    (tmp_path / "u.item").write_text(U_ITEM_SAMPLE, encoding="latin-1")

    df = parse_movies(tmp_path, "100k")

    assert list(df.columns) == ["movieId", "title", "genres"]
    assert df.shape[0] == 3

    toy_story = df.loc[df["movieId"] == 1].iloc[0]
    assert toy_story["title"] == "Toy Story (1995)"
    assert toy_story["genres"] == "Animation|Children's|Comedy"

    se7en = df.loc[df["movieId"] == 2].iloc[0]
    assert se7en["genres"] == "Crime|Drama|Thriller"


def test_parse_movies_100k_unknown_genre_uses_placeholder_not_empty_string(tmp_path: Path):
    # Une chaine vide serait relue comme NaN par pandas.read_csv (cote nous
    # ET cote backend) et afficherait litteralement "nan" dans la demo.
    (tmp_path / "u.item").write_text(U_ITEM_SAMPLE, encoding="latin-1")

    df = parse_movies(tmp_path, "100k")

    mystery = df.loc[df["movieId"] == 3].iloc[0]
    assert mystery["genres"] == "Genre inconnu"
    assert mystery["genres"] != ""


def test_parse_movies_100k_raises_if_file_missing(tmp_path: Path):
    try:
        parse_movies(tmp_path, "100k")
    except FileNotFoundError:
        return
    raise AssertionError("parse_movies aurait du lever FileNotFoundError")
