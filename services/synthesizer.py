"""
Service de synthèse qualitative par macro-catégorie.

Prend les résumés et sentiments d'une catégorie et demande au LLM d'en extraire
les points forts, les irritants, et une synthèse globale rédigée.
"""

import json
from typing import Optional
from models.schemas import CategorySynthesis, VerbatimAnalysis
from services.llm_client import OllamaClient, _extract_json


def build_synthesis_prompt(macro_theme: str, reviews: list[dict]) -> str:
    """Construit le prompt pour demander au LLM de synthétiser une catégorie."""
    # Formater les reviews pour le prompt
    formatted_reviews = []
    for r in reviews:
        sentiment = r.get("sentiment", "neutre")
        summary = r.get("summary", "")
        formatted_reviews.append(f"- [{sentiment}] {summary}")
        
    reviews_list = "\n".join(formatted_reviews)
    
    return f"""Tu es un analyste de l'expérience client spécialisé dans l'analyse de satisfaction.
Voici une liste d'extraits d'avis clients et leurs sentiments associés pour la macro-thématique : "{macro_theme}".

Avis clients :
{reviews_list}

Ton objectif est de rédiger une synthèse qualitative structurée de cette catégorie en identifiant les idées récurrentes.

Réponds UNIQUEMENT avec un JSON valide respectant cette structure exacte :
{{
  "macro_theme": "{macro_theme}",
  "positive_points": [
    "Point fort 1 (ex: clarté des tarifs)",
    "Point fort 2 (ex: rapidité d'exécution)"
  ],
  "negative_points": [
    "Point faible 1 (ex: bugs de connexion)",
    "Point faible 2 (ex: attente téléphonique trop longue)"
  ],
  "global_synthesis": "Synthèse rédigée en français (2 à 4 phrases maximum) expliquant la tendance globale et les arbitrages des clients."
}}

RÈGLES CRITIQUES :
1. Les points positifs et négatifs doivent être clairs, courts et fidèles aux avis fournis.
2. S'il n'y a aucun point positif ou négatif représenté dans les avis, laisse la liste vide `[]`.
3. La synthèse globale doit obligatoirement être rédigée en français.
4. Ne rien écrire avant ou après le JSON.
"""


class CategorySynthesizer:
    """Génère la synthèse qualitative d'une catégorie à partir de ses verbatims."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    async def synthesize(self, macro_theme: str, analyses: list[VerbatimAnalysis]) -> CategorySynthesis:
        """Demande au LLM de générer la synthèse qualitative d'une macro-catégorie."""
        if not analyses:
            return CategorySynthesis(
                macro_theme=macro_theme,
                positive_points=[],
                negative_points=[],
                global_synthesis="Aucun avis disponible pour cette catégorie."
            )
            
        reviews_data = [
            {"sentiment": a.sentiment, "summary": a.summary} 
            for a in analyses
        ]
        
        prompt = build_synthesis_prompt(macro_theme, reviews_data)
        raw = await self.client._call_api(prompt)
        
        if not raw:
            raise ValueError(f"Aucune réponse du LLM pour la synthèse de {macro_theme}")
            
        parsed = _extract_json(raw)
        if not parsed:
            raise ValueError(f"Impossible de parser la réponse du LLM pour {macro_theme} : {raw[:200]}")
            
        # Validation Pydantic
        return CategorySynthesis.model_validate(parsed)
