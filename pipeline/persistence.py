"""
Persistance — flux de sortie séparés pour résultats et erreurs.

Format JSONL : un objet JSON par ligne, facile à lire/écrire incrémentalement.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import PROCESSED_DIR
from models.schemas import VerbatimAnalysis


def _get_timestamp() -> str:
    """Timestamp pour nommer les fichiers de sortie."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(
    analyses: list[VerbatimAnalysis],
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Sauvegarde les résultats valides en JSONL.

    Returns:
        Chemin du fichier créé.
    """
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"results_{run_id or _get_timestamp()}.jsonl"
    path = output_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        for analysis in analyses:
            f.write(analysis.model_dump_json() + "\n")

    print(f"💾 {len(analyses)} résultats → {path}")
    return path


def save_errors(
    errors: list[dict],
    output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Path:
    """Sauvegarde les erreurs en JSONL.

    Returns:
        Chemin du fichier créé.
    """
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"errors_{run_id or _get_timestamp()}.jsonl"
    path = output_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        for error in errors:
            f.write(json.dumps(error, ensure_ascii=False, default=str) + "\n")

    print(f"⚠️  {len(errors)} erreurs → {path}")
    return path


def append_results(
    analyses: list[VerbatimAnalysis],
    path: Path,
) -> None:
    """Ajoute des résultats à un fichier JSONL existant (mode append)."""
    with open(path, "a", encoding="utf-8") as f:
        for analysis in analyses:
            f.write(analysis.model_dump_json() + "\n")


def load_results(path: Path) -> list[VerbatimAnalysis]:
    """Recharge des résultats depuis un fichier JSONL."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(VerbatimAnalysis.model_validate_json(line))
    return results
