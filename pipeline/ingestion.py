"""
Couche d'ingestion — normalise toute source vers list[VerbatimInput].

Supporte : CSV, JSONL, DataFrames pandas.
Auto-détection des colonnes texte courantes.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from models.schemas import VerbatimInput


# Colonnes texte courantes dans les datasets connus
TEXT_COLUMN_CANDIDATES = [
    "text", "review", "review_body", "review_text",
    "content", "comment", "verbatim", "avis",
]

ID_COLUMN_CANDIDATES = [
    "id", "review_id", "verbatim_id", "index",
]


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Détecte automatiquement la bonne colonne parmi les candidats."""
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def load_csv(
    path: str | Path,
    text_col: Optional[str] = None,
    id_col: Optional[str] = None,
    source: str = "csv",
    lang: str = "fr",
    limit: Optional[int] = None,
) -> list[VerbatimInput]:
    """Charge un CSV et le normalise en VerbatimInput.

    Args:
        path: chemin vers le fichier CSV
        text_col: nom de la colonne texte (auto-détecté si None)
        id_col: nom de la colonne ID (généré si None)
        source: label de traçabilité
        lang: langue par défaut
        limit: nombre max de verbatims à charger
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_csv(path, nrows=limit)
    print(f"📂 Chargé {len(df)} lignes depuis {path.name}")
    print(f"   Colonnes: {list(df.columns)}")

    # Auto-détection de la colonne texte
    if text_col is None:
        text_col = _detect_column(df, TEXT_COLUMN_CANDIDATES)
        if text_col is None:
            raise ValueError(
                f"Impossible de détecter la colonne texte. "
                f"Colonnes disponibles: {list(df.columns)}. "
                f"Spécifiez text_col explicitement."
            )
    print(f"   Colonne texte: '{text_col}'")

    # Auto-détection ou génération des IDs
    if id_col is None:
        id_col = _detect_column(df, ID_COLUMN_CANDIDATES)
    if id_col:
        print(f"   Colonne ID: '{id_col}'")

    # Conversion en VerbatimInput
    verbatims = []
    for idx, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text or text == "nan":
            continue

        vid = str(row[id_col]) if id_col else f"{source}_{idx}"
        verbatims.append(VerbatimInput(
            id=vid,
            text=text,
            source=source,
            lang=lang,
        ))

    print(f"   ✅ {len(verbatims)} verbatims valides (sur {len(df)} lignes)")
    return verbatims


def load_jsonl(
    path: str | Path,
    text_field: str = "text",
    id_field: str = "id",
    source: str = "jsonl",
    lang: str = "fr",
    limit: Optional[int] = None,
) -> list[VerbatimInput]:
    """Charge un fichier JSONL et le normalise en VerbatimInput."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    df = pd.read_json(path, lines=True, nrows=limit)
    print(f"📂 Chargé {len(df)} lignes depuis {path.name}")

    verbatims = []
    for idx, row in df.iterrows():
        text = str(row.get(text_field, "")).strip()
        if not text or text == "nan":
            continue

        vid = str(row.get(id_field, f"{source}_{idx}"))
        verbatims.append(VerbatimInput(
            id=vid,
            text=text,
            source=source,
            lang=lang,
        ))

    print(f"   ✅ {len(verbatims)} verbatims valides")
    return verbatims


def from_dataframe(
    df: pd.DataFrame,
    text_col: str,
    id_col: Optional[str] = None,
    source: str = "dataframe",
    lang: str = "fr",
    limit: Optional[int] = None,
) -> list[VerbatimInput]:
    """Convertit un DataFrame pandas en VerbatimInput."""
    if limit:
        df = df.head(limit)

    verbatims = []
    for idx, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text or text == "nan":
            continue

        vid = str(row[id_col]) if id_col else f"{source}_{idx}"
        verbatims.append(VerbatimInput(
            id=vid,
            text=text,
            source=source,
            lang=lang,
        ))

    print(f"   ✅ {len(verbatims)} verbatims depuis DataFrame")
    return verbatims
