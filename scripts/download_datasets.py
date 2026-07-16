"""
Télécharge et prépare les datasets open source pour tester le pipeline.

Datasets :
- Allociné (FR, 200k reviews de films)
- Bank Reviews (EN, 20k avis bancaires)
- Amazon FR (FR, reviews multilingues)

Chaque dataset est converti en CSV normalisé : id, text, source, lang
"""

import sys
from pathlib import Path

# Ajouter le root du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import RAW_DIR


def download_allocine(limit: int = 1000) -> Path:
    """Télécharge un échantillon d'Allociné (reviews de films FR)."""
    from datasets import load_dataset

    print(f"\n📥 Téléchargement Allociné (limit={limit})...")
    ds = load_dataset("tblard/allocine", split=f"train[:{limit}]")
    df = ds.to_pandas()

    # Normaliser
    out = pd.DataFrame({
        "id": [f"allocine_{i}" for i in range(len(df))],
        "text": df["review"],
        "source": "allocine",
        "lang": "fr",
    })

    path = RAW_DIR / "allocine_sample.csv"
    out.to_csv(path, index=False)
    print(f"   ✅ {len(out)} reviews → {path}")
    return path


def download_bank_reviews(limit: int = 1000) -> Path:
    """Télécharge un échantillon de Bank Reviews (avis bancaires EN)."""
    from datasets import load_dataset

    print(f"\n📥 Téléchargement Bank Reviews (limit={limit})...")
    ds = load_dataset("UniqueData/customers-reviews-on-banks", split=f"train[:{limit}]")
    df = ds.to_pandas()

    # Chercher la colonne texte
    text_col = None
    for candidate in ["review", "text", "review_text", "Text", "Review"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        print(f"   ⚠️  Colonnes disponibles: {list(df.columns)}")
        # Prendre la colonne avec le plus de texte
        text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
        print(f"   → Utilisation de la colonne '{text_col}' (plus long texte moyen)")

    out = pd.DataFrame({
        "id": [f"bank_{i}" for i in range(len(df))],
        "text": df[text_col],
        "source": "bank_reviews",
        "lang": "en",
    })

    path = RAW_DIR / "bank_reviews_sample.csv"
    out.to_csv(path, index=False)
    print(f"   ✅ {len(out)} reviews → {path}")
    return path


def download_amazon_fr(limit: int = 1000) -> Path:
    """Télécharge un échantillon d'Amazon Reviews FR."""
    from datasets import load_dataset

    print(f"\n📥 Téléchargement Amazon FR (limit={limit})...")
    ds = load_dataset("SetFit/amazon_reviews_multi_fr", split=f"train[:{limit}]")
    df = ds.to_pandas()

    text_col = None
    for candidate in ["text", "review_body", "review_text"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        print(f"   ⚠️  Colonnes disponibles: {list(df.columns)}")
        text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
        print(f"   → Utilisation de la colonne '{text_col}'")

    out = pd.DataFrame({
        "id": [f"amazon_fr_{i}" for i in range(len(df))],
        "text": df[text_col],
        "source": "amazon_fr",
        "lang": "fr",
    })

    path = RAW_DIR / "amazon_fr_sample.csv"
    out.to_csv(path, index=False)
    print(f"   ✅ {len(out)} reviews → {path}")
    return path


def create_test_sample() -> Path:
    """Crée un petit fichier de test avec des verbatims hardcodés."""
    verbatims = [
        {"id": "test_001", "text": "Le service est trop lent et les frais sont exorbitants. Je n'en peux plus.", "source": "test", "lang": "fr"},
        {"id": "test_002", "text": "Excellent accueil en agence, le conseiller a pris le temps de m'expliquer toutes les options.", "source": "test", "lang": "fr"},
        {"id": "test_003", "text": "L'application mobile plante systématiquement quand j'essaie de faire un virement.", "source": "test", "lang": "fr"},
        {"id": "test_004", "text": "The customer service representative was incredibly helpful and resolved my issue in minutes.", "source": "test", "lang": "en"},
        {"id": "test_005", "text": "J'ai attendu 45 minutes au téléphone pour rien, personne n'a su me répondre.", "source": "test", "lang": "fr"},
        {"id": "test_006", "text": "Carte bloquée à l'étranger sans prévenir, j'étais bloqué sans moyen de paiement.", "source": "test", "lang": "fr"},
        {"id": "test_007", "text": "Le nouveau design de l'app est très intuitif, bravo à l'équipe.", "source": "test", "lang": "fr"},
        {"id": "test_008", "text": "Impossible de joindre le service client depuis 3 jours, les lignes sont toujours occupées.", "source": "test", "lang": "fr"},
    ]

    df = pd.DataFrame(verbatims)
    path = RAW_DIR / "test_sample.csv"
    df.to_csv(path, index=False)
    print(f"\n📝 Fichier test créé: {path} ({len(verbatims)} verbatims)")
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Télécharge les datasets")
    parser.add_argument("--dataset", choices=["all", "allocine", "bank", "amazon", "test"],
                        default="test", help="Dataset à télécharger")
    parser.add_argument("--limit", type=int, default=1000, help="Nombre de reviews")
    args = parser.parse_args()

    if args.dataset in ("all", "test"):
        create_test_sample()

    if args.dataset in ("all", "allocine"):
        download_allocine(args.limit)

    if args.dataset in ("all", "bank"):
        download_bank_reviews(args.limit)

    if args.dataset in ("all", "amazon"):
        download_amazon_fr(args.limit)

    print("\n✅ Terminé !")
