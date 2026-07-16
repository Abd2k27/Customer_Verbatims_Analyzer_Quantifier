"""
Couche de validation — règles métier post-LLM.

Séparation stricte : résultats valides vs erreurs (jamais mélangés).
"""

from models.schemas import VerbatimInput, VerbatimAnalysis, BatchAnalysisResult


class ValidationError(Exception):
    """Erreur de validation métier (pas une erreur Pydantic)."""
    pass


def validate_batch_result(
    result: BatchAnalysisResult,
    expected_verbatims: list[VerbatimInput],
) -> tuple[list[VerbatimAnalysis], list[dict]]:
    """Valide un résultat de batch et sépare les bons résultats des erreurs.

    Returns:
        (valid_analyses, errors) — jamais mélangés.
    """
    valid = []
    errors = []
    expected_ids = {v.id for v in expected_verbatims}

    # Vérification du nombre d'items
    if len(result.analyses) != len(expected_verbatims):
        errors.append({
            "type": "count_mismatch",
            "expected": len(expected_verbatims),
            "got": len(result.analyses),
            "detail": f"Le LLM a retourné {len(result.analyses)} analyses "
                      f"au lieu de {len(expected_verbatims)}",
        })

    seen_ids = set()
    for analysis in result.analyses:
        item_errors = []

        # ID dupliqué
        if analysis.verbatim_id in seen_ids:
            item_errors.append(f"ID dupliqué: {analysis.verbatim_id}")
        seen_ids.add(analysis.verbatim_id)

        # ID inconnu
        if analysis.verbatim_id not in expected_ids:
            item_errors.append(
                f"ID inconnu: {analysis.verbatim_id} "
                f"(attendus: {expected_ids})"
            )

        # Thème vide ou trop long
        if not analysis.theme or not analysis.theme.strip():
            item_errors.append("Thème vide")
        elif len(analysis.theme.split()) > 5:
            item_errors.append(
                f"Thème trop long ({len(analysis.theme.split())} mots): "
                f"'{analysis.theme}'"
            )

        # Résumé trop court
        if not analysis.summary or len(analysis.summary.strip()) < 5:
            item_errors.append(f"Résumé trop court: '{analysis.summary}'")

        if item_errors:
            errors.append({
                "type": "item_validation",
                "verbatim_id": analysis.verbatim_id,
                "issues": item_errors,
                "data": analysis.model_dump(),
            })
        else:
            valid.append(analysis)

    # Verbatims manquants (le LLM les a sautés)
    returned_ids = {a.verbatim_id for a in result.analyses}
    missing = expected_ids - returned_ids
    if missing:
        errors.append({
            "type": "missing_verbatims",
            "missing_ids": list(missing),
            "detail": f"{len(missing)} verbatim(s) sauté(s) par le LLM",
        })

    return valid, errors
