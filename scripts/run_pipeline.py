"""
Point d'entrée principal — CLI pour lancer le pipeline d'analyse.

Usage:
    python scripts/run_pipeline.py --input data/raw/test_sample.csv --limit 5
    python scripts/run_pipeline.py --input data/raw/allocine_sample.csv --batch-size 10
    python scripts/run_pipeline.py --input data/raw/allocine_sample.csv --resume
    python scripts/run_pipeline.py --test  # test rapide de connectivité
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Ajouter le root du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import BATCH_SIZE, MAX_RETRIES
from services.llm_client import OllamaClient
from pipeline.ingestion import load_csv
from pipeline.orchestrator import PipelineOrchestrator


async def test_connectivity():
    """Test rapide de connectivité avec Ollama Cloud."""
    print("\n🔌 Test de connectivité Ollama Cloud...")
    client = OllamaClient()
    ok = await client.test_connectivity()
    if ok:
        print("✅ Tout est bon, l'API répond !")
    else:
        print("❌ Problème de connexion — vérifie l'URL et l'API key dans .env")
    return ok


async def run_pipeline(
    input_path: str,
    batch_size: int,
    max_retries: int,
    limit: int | None,
    resume: bool,
    text_col: str | None,
    source: str | None,
    lang: str,
    run_id: str | None,
):
    """Lance le pipeline complet."""
    # 1. Ingestion
    path = Path(input_path)
    src = source or path.stem.split("_")[0]

    print(f"\n📥 Chargement des verbatims depuis {path.name}...")
    verbatims = load_csv(
        path,
        text_col=text_col,
        source=src,
        lang=lang,
        limit=limit,
    )

    if not verbatims:
        print("❌ Aucun verbatim trouvé !")
        return

    # 2. Orchestration
    client = OllamaClient()
    orchestrator = PipelineOrchestrator(
        client=client,
        batch_size=batch_size,
        max_retries=max_retries,
        run_id=run_id,
    )

    results, errors = await orchestrator.run(
        verbatims,
        resume=resume,
    )

    # 3. Affichage des premiers résultats
    if results:
        print(f"\n📋 Aperçu des {min(5, len(results))} premiers résultats :")
        print(f"{'─'*80}")
        for r in results[:5]:
            print(f"  [{r.verbatim_id}]")
            print(f"    🏷️  Thème:  {r.theme}")
            print(f"    📝 Résumé: {r.summary}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline d'analyse de sentiment — verbatims clients"
    )
    parser.add_argument("--test", action="store_true",
                        help="Test de connectivité uniquement")
    parser.add_argument("--input", "-i", type=str,
                        help="Fichier CSV d'entrée")
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE,
                        help=f"Taille des batches (défaut: {BATCH_SIZE})")
    parser.add_argument("--max-retries", "-r", type=int, default=MAX_RETRIES,
                        help=f"Nombre de retries par batch (défaut: {MAX_RETRIES})")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Nombre max de verbatims à traiter")
    parser.add_argument("--resume", action="store_true",
                        help="Reprendre un run interrompu")
    parser.add_argument("--run-id", type=str, default=None,
                        help="ID du run (pour reprise)")
    parser.add_argument("--text-col", type=str, default=None,
                        help="Nom de la colonne texte dans le CSV")
    parser.add_argument("--source", type=str, default=None,
                        help="Label de source pour la traçabilité")
    parser.add_argument("--lang", type=str, default="fr",
                        help="Langue par défaut (défaut: fr)")

    args = parser.parse_args()

    if args.test:
        asyncio.run(test_connectivity())
        return

    if not args.input:
        parser.error("--input est requis (ou utilisez --test)")

    if not Path(args.input).exists():
        parser.error(f"Fichier introuvable: {args.input}")

    asyncio.run(run_pipeline(
        input_path=args.input,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        limit=args.limit,
        resume=args.resume,
        text_col=args.text_col,
        source=args.source,
        lang=args.lang,
        run_id=args.run_id,
    ))


if __name__ == "__main__":
    main()
