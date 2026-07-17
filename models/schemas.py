"""
Modèles Pydantic pour le pipeline d'analyse de sentiment.

Contrat de données non-négociable : tout le pipeline respecte ces schémas.
"""

from pydantic import BaseModel, Field


from typing import Literal, Optional


class VerbatimInput(BaseModel):
    """Format normalisé d'entrée — toute source passe par ce format."""
    id: str
    text: str
    source: str = "unknown"  # traçabilité (allocine, bank_reviews, amazon_fr)
    lang: str = "fr"         # langue détectée ou déclarée


class VerbatimAnalysis(BaseModel):
    """Résultat d'analyse d'un seul verbatim."""
    verbatim_id: str
    theme: str = Field(
        description="Thématique principale, en 1 à 3 mots, formulée librement"
    )
    summary: str = Field(
        description="Résumé en une phrase courte",
        max_length=150
    )
    sentiment: Literal["positif", "négatif", "neutre"] = Field(
        description="Sentiment associé au verbatim : positif, négatif, ou neutre"
    )
    texte_original: Optional[str] = Field(default=None, description="Texte original du verbatim")
    source: Optional[str] = Field(default=None, description="Source d'origine du verbatim")
    macro_theme: Optional[str] = Field(default=None, description="Catégorie macro après harmonisation")


class BatchAnalysisResult(BaseModel):
    """Résultat d'analyse d'un batch de verbatims."""
    analyses: list[VerbatimAnalysis]


class CategorySynthesis(BaseModel):
    """Synthèse qualitative par macro-catégorie."""
    macro_theme: str
    positive_points: list[str] = Field(
        description="Liste des points de satisfaction récurrents (2 à 5 points)"
    )
    negative_points: list[str] = Field(
        description="Liste des points d'insatisfaction ou d'irritants récurrents (2 à 5 points)"
    )
    global_synthesis: str = Field(
        description="Résumé synthétique rédigé en français (2 à 4 phrases)"
    )
