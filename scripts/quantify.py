"""
Script de quantification et d'analyse des résultats.

Réalise deux étapes :
1. Harmonisation sémantique des thématiques libres via LLM.
2. Calcul des taux de sentiments globaux et par thématique.
"""

import sys
import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime

# Ajouter le root du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROCESSED_DIR
from pipeline.persistence import load_results
from services.harmonizer import ThemeHarmonizer


def display_report(
    total: int,
    global_sentiment: dict,
    theme_distribution: dict,
    categories: list[str],
    mapping: dict,
):
    """Affiche un joli rapport textuel dans le terminal."""
    print("\n" + "="*80)
    print(f"📊 RAPPORT DE QUANTIFICATION ET D'ANALYSE DES SENTIMENTS")
    print("="*80)
    print(f"Total verbatims analysés : {total}\n")

    # 1. Sentiments Globaux
    print("📈 RÉPARTITION GLOBALE DES SENTIMENTS :")
    print("─"*40)
    for sent, count in global_sentiment.items():
        percentage = (count / total) * 100
        bar = "█" * int(percentage // 5)
        print(f"  {sent:<10} : {percentage:>5.1f}% ({count:>3})  {bar}")
    print()

    # 2. Répartition par thématique macro
    print("🏷️  RÉPARTITION PAR CATÉGORIE HARMONISÉE (ET SENTIMENT) :")
    print("─"*80)
    # Trier les catégories par volume décroissant
    sorted_cats = sorted(
        theme_distribution.keys(),
        key=lambda c: theme_distribution[c]["total"],
        reverse=True,
    )

    for cat in sorted_cats:
        data = theme_distribution[cat]
        cat_total = data["total"]
        cat_percentage = (cat_total / total) * 100
        print(f"🔹 {cat:<30} | Total: {cat_total:>3} ({cat_percentage:>5.1f}%)")

        # Distribution du sentiment dans cette catégorie
        for sent in ["positif", "négatif", "neutre"]:
            count = data.get(sent, 0)
            if count > 0:
                sent_percentage = (count / cat_total) * 100
                bar = "░" * int(sent_percentage // 10)
                print(f"    - {sent:<10} : {sent_percentage:>5.1f}% ({count:>2}) {bar}")
        print("─"*80)


async def main():
    parser = argparse.ArgumentParser(
        description="Quantification et harmonisation des résultats d'analyse."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Chemin vers le fichier de résultats JSONL (ex: data/processed/results_*.jsonl)"
    )
    parser.add_argument(
        "--max-categories", "-c", type=int, default=5,
        help="Nombre maximum de catégories thématiques harmonisées"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Fichier introuvable : {args.input}")
        return

    # 1. Charger les résultats
    analyses = load_results(input_path)
    if not analyses:
        print("❌ Aucun résultat trouvé dans le fichier.")
        return

    total = len(analyses)
    print(f"📖 Chargement de {total} verbatims analysés...")

    # 2. Récupérer les thèmes uniques
    unique_themes = sorted(list({a.theme for a in analyses if a.theme}))
    print(f"🔍 Trouvé {len(unique_themes)} thèmes libres uniques.")

    # 3. Harmoniser via LLM
    print("🧠 Harmonisation des thèmes en cours (appel LLM)...")
    harmonizer = ThemeHarmonizer()
    try:
        harmonization = await harmonizer.harmonize(
            unique_themes,
            max_categories=args.max_categories
        )
        print(f"✅ Thèmes regroupés en {len(harmonization.categories)} catégories :")
        for cat in harmonization.categories:
            print(f"   - {cat}")
    except Exception as e:
        print(f"❌ Échec de l'harmonisation : {e}")
        return

    # 4. Calculer les statistiques
    global_sentiment = {"positif": 0, "négatif": 0, "neutre": 0}
    theme_distribution = {}

    # Initialiser la structure de distribution pour chaque catégorie macro
    for cat in harmonization.categories:
        theme_distribution[cat] = {"total": 0, "positif": 0, "négatif": 0, "neutre": 0}

    # Analyser chaque verbatim
    enriched_results = []
    for analysis in analyses:
        sentiment = analysis.sentiment
        orig_theme = analysis.theme

        # Incrémenter sentiment global
        global_sentiment[sentiment] = global_sentiment.get(sentiment, 0) + 1

        # Trouver la catégorie harmonisée
        macro_cat = harmonization.mapping.get(orig_theme, "Autre")
        if macro_cat not in theme_distribution:
            theme_distribution[macro_cat] = {"total": 0, "positif": 0, "négatif": 0, "neutre": 0}

        theme_distribution[macro_cat]["total"] += 1
        theme_distribution[macro_cat][sentiment] += 1

        # Enrichir l'objet de résultat pour la sauvegarde
        enriched_item = analysis.model_dump()
        enriched_item["macro_theme"] = macro_cat
        enriched_results.append(enriched_item)

    # 5. Affichage du rapport
    display_report(
        total,
        global_sentiment,
        theme_distribution,
        harmonization.categories,
        harmonization.mapping,
    )

    # 6. Sauvegarder les résultats enrichis et le rapport
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = PROCESSED_DIR / f"quantification_report_{timestamp}.json"
    enriched_path = PROCESSED_DIR / f"enriched_results_{timestamp}.jsonl"

    # Sauvegarde des résultats enrichis
    with open(enriched_path, "w", encoding="utf-8") as f:
        for item in enriched_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Sauvegarde du rapport
    report_data = {
        "metadata": {
            "source_file": str(input_path),
            "timestamp": datetime.now().isoformat(),
            "total_verbatims": total,
        },
        "global_sentiment": global_sentiment,
        "theme_distribution": theme_distribution,
        "mapping": harmonization.mapping,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Résultats enrichis sauvegardés dans : {enriched_path}")
    print(f"📝 Rapport JSON complet sauvegardé dans : {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
