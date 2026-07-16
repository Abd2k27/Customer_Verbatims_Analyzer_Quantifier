"""
Orchestrateur — batching, checkpointing sur disque, reprise idempotente.

Si le run plante au batch 47 sur 200, on peut reprendre à 47 sans refaire les 46 premiers.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from config.settings import BATCH_SIZE, MAX_RETRIES, CHECKPOINT_DIR
from models.schemas import VerbatimInput, VerbatimAnalysis
from services.llm_client import OllamaClient
from services.validator import validate_batch_result
from pipeline.persistence import append_results, save_errors


class PipelineOrchestrator:
    """Orchestre le traitement par batch avec checkpointing."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        batch_size: int = BATCH_SIZE,
        max_retries: int = MAX_RETRIES,
        run_id: Optional[str] = None,
    ):
        self.client = client or OllamaClient()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Fichiers de checkpoint
        self._checkpoint_dir = CHECKPOINT_DIR / self.run_id
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_file = self._checkpoint_dir / "progress.json"

    def _load_checkpoint(self) -> set[int]:
        """Charge les indices des batches déjà traités."""
        if self._checkpoint_file.exists():
            with open(self._checkpoint_file, "r") as f:
                data = json.load(f)
                completed = set(data.get("completed_batches", []))
                print(f"📋 Checkpoint chargé: {len(completed)} batches déjà traités")
                return completed
        return set()

    def _save_checkpoint(self, completed: set[int], total: int) -> None:
        """Sauvegarde l'état du pipeline sur disque."""
        with open(self._checkpoint_file, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "completed_batches": sorted(completed),
                "total_batches": total,
                "last_update": datetime.now().isoformat(),
            }, f, indent=2)

    def _make_batches(
        self, verbatims: list[VerbatimInput]
    ) -> list[tuple[int, list[VerbatimInput]]]:
        """Découpe en batches indexés."""
        batches = []
        for i in range(0, len(verbatims), self.batch_size):
            batch = verbatims[i:i + self.batch_size]
            batches.append((i // self.batch_size, batch))
        return batches

    async def run(
        self,
        verbatims: list[VerbatimInput],
        output_path: Optional[Path] = None,
        resume: bool = True,
        on_progress: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> tuple[list[VerbatimAnalysis], list[dict]]:
        """Lance le pipeline complet.

        Args:
            verbatims: liste de verbatims à traiter
            output_path: fichier JSONL de sortie (créé automatiquement si None)
            resume: si True, reprend depuis le checkpoint existant
            on_progress: callback optionnel appelé après chaque batch (batch_idx, total_batches, completed_count, error_count)

        Returns:
            (all_valid_results, all_errors)
        """
        batches = self._make_batches(verbatims)
        total_batches = len(batches)
        print(f"\n{'='*60}")
        print(f"🚀 Pipeline démarré — run_id: {self.run_id}")
        print(f"   {len(verbatims)} verbatims → {total_batches} batches "
              f"(taille {self.batch_size})")
        print(f"{'='*60}\n")

        # Charger le checkpoint si reprise
        completed = self._load_checkpoint() if resume else set()

        # Fichier de sortie
        if output_path is None:
            from config.settings import PROCESSED_DIR
            output_path = PROCESSED_DIR / f"results_{self.run_id}.jsonl"

        all_results: list[VerbatimAnalysis] = []
        all_errors: list[dict] = []

        for batch_idx, batch in batches:
            # Skip si déjà traité
            if batch_idx in completed:
                print(f"  ⏭️  Batch {batch_idx + 1}/{total_batches} — déjà traité")
                continue

            batch_ids = [v.id for v in batch]
            print(f"\n  📦 Batch {batch_idx + 1}/{total_batches} "
                  f"({len(batch)} verbatims: {batch_ids[0]}..{batch_ids[-1]})")

            try:
                # Appel LLM
                result = await self.client.analyze_batch(
                    batch,
                    max_retries=self.max_retries,
                )

                # Validation métier
                valid, errors = validate_batch_result(result, batch)

                if valid:
                    all_results.extend(valid)
                    append_results(valid, output_path)
                    print(f"  ✅ {len(valid)} analyses valides")

                if errors:
                    all_errors.extend(errors)
                    for err in errors:
                        print(f"  ⚠️  {err.get('type', 'unknown')}: "
                              f"{err.get('detail', err.get('issues', ''))}")

                # Marquer comme traité même avec des erreurs partielles
                completed.add(batch_idx)
                self._save_checkpoint(completed, total_batches)

            except ValueError as e:
                # Échec complet du batch (toutes tentatives épuisées)
                error = {
                    "type": "batch_failure",
                    "batch_index": batch_idx,
                    "verbatim_ids": batch_ids,
                    "error": str(e),
                }
                all_errors.append(error)
                print(f"  ❌ Batch échoué: {e}")

                # On ne le marque PAS comme completed → rejouable
                self._save_checkpoint(completed, total_batches)

            except Exception as e:
                error = {
                    "type": "unexpected_error",
                    "batch_index": batch_idx,
                    "verbatim_ids": batch_ids,
                    "error": str(e),
                }
                all_errors.append(error)
                print(f"  💥 Erreur inattendue: {e}")
                self._save_checkpoint(completed, total_batches)

            # Appel du callback de progression
            if on_progress:
                try:
                    on_progress(batch_idx + 1, total_batches, len(all_results), len(all_errors))
                except Exception as cb_err:
                    print(f"  ⚠️  Erreur dans le callback on_progress: {cb_err}")

        # Résumé final
        print(f"\n{'='*60}")
        print(f"📊 Pipeline terminé — run_id: {self.run_id}")
        print(f"   ✅ {len(all_results)} analyses valides")
        print(f"   ⚠️  {len(all_errors)} erreurs")
        print(f"   📁 Résultats: {output_path}")
        print(f"{'='*60}\n")

        # Sauvegarder les erreurs
        if all_errors:
            from config.settings import PROCESSED_DIR
            save_errors(all_errors, PROCESSED_DIR, self.run_id)

        return all_results, all_errors
