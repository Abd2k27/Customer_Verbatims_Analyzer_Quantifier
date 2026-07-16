"""
Service d'harmonisation des thématiques libres.

Regroupe les thèmes libres et fragmentés (ex: "frais bancaires", "frais trop élevés")
en catégories macro définies dynamiquement par le LLM ou par l'utilisateur.
"""

import json
from typing import Optional
from pydantic import BaseModel, Field

from config.settings import OLLAMA_MODEL
from services.llm_client import OllamaClient, _extract_json


class HarmonizationResult(BaseModel):
    """Schéma de sortie pour l'harmonisation des thèmes."""
    categories: list[str] = Field(
        description="Liste des grandes catégories thématiques retenues"
    )
    mapping: dict[str, str] = Field(
        description="Dictionnaire associant CHAQUE thème d'origine à une des grandes catégories"
    )


def build_harmonization_prompt(unique_themes: list[str], max_categories: int = 5) -> str:
    """Construit le prompt pour demander au LLM d'harmoniser les thèmes."""
    themes_list = "\n".join(f"- {theme}" for theme in unique_themes)
    return f"""Tu es un expert en analyse de données et classification de verbatims clients.
On te donne une liste de thématiques libres extraites de commentaires clients. Cette liste est fragmentée (plusieurs manières de dire la même chose).

Ton rôle est de :
1. Regrouper ces thèmes libres en un nombre restreint de catégories macro cohérentes (maximum {max_categories} catégories).
2. Associer CHAQUE thème d'origine de la liste à l'une de ces catégories macro.

Voici la liste des thèmes d'origine à traiter :
{themes_list}

Réponds UNIQUEMENT avec un JSON valide respectant cette structure exacte :
{{
  "categories": ["Nom Catégorie 1", "Nom Catégorie 2", ...],
  "mapping": {{
    "Thème d'origine A": "Nom Catégorie 1",
    "Thème d'origine B": "Nom Catégorie 2",
    ...
  }}
}}

RÈGLES CRITIQUES :
1. CHAQUE thème d'origine de la liste ci-dessus doit figurer comme clé dans le dictionnaire "mapping".
2. La valeur associée à chaque clé doit faire partie de la liste "categories".
3. Ne pas inventer de thèmes d'origine qui ne sont pas dans la liste.
4. Les noms de catégories macro doivent être clairs, concis (1 à 3 mots) et englobants.
"""


class ThemeHarmonizer:
    """Harmonise les thématiques libres via LLM."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()

    async def harmonize(
        self,
        unique_themes: list[str],
        max_categories: int = 5,
    ) -> HarmonizationResult:
        """Demande au LLM de générer un mapping d'harmonisation."""
        if not unique_themes:
            return HarmonizationResult(categories=[], mapping={})

        prompt = build_harmonization_prompt(unique_themes, max_categories)
        raw = await self.client._call_api(prompt)

        if not raw:
            raise ValueError("Aucune réponse du LLM pour l'harmonisation")

        parsed = _extract_json(raw)
        if not parsed:
            raise ValueError(f"Impossible d'extraire le JSON d'harmonisation de: {raw[:200]}")

        # Validation Pydantic
        result = HarmonizationResult.model_validate(parsed)

        # Post-validation de sécurité : s'assurer que chaque thème d'origine a un mapping
        for theme in unique_themes:
            if theme not in result.mapping:
                # Fallback : s'associer à une catégorie "Autre" ou la première dispo
                fallback_cat = result.categories[0] if result.categories else "Autre"
                if "Autre" not in result.categories:
                    result.categories.append("Autre")
                    fallback_cat = "Autre"
                result.mapping[theme] = fallback_cat

        return result
