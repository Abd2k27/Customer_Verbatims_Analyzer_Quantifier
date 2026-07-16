"""
Client LLM Ollama Cloud — wrapper httpx async.

Pattern identique à findme-app/models/llm_extractor.py :
- httpx.AsyncClient pour les appels HTTP
- Header Authorization: Bearer {api_key}
- Endpoint /api/chat avec format: "json"
- Parsing robuste du JSON retourné (regex fallback)
- Retry avec backoff exponentiel
"""

import asyncio
import json
import re
import httpx
from typing import Optional

from config.settings import (
    OLLAMA_MODEL, OLLAMA_URL, OLLAMA_API_KEY,
    OLLAMA_TIMEOUT, TEMPERATURE
)
from models.schemas import VerbatimInput, VerbatimAnalysis, BatchAnalysisResult


def build_prompt(verbatims: list[VerbatimInput]) -> str:
    """Construit le prompt pour un batch de verbatims."""
    lines = "\n".join(f"[{v.id}] {v.text}" for v in verbatims)
    return f"""Tu es un analyste qui étudie des verbatims clients.
Pour chaque verbatim ci-dessous :
1. Identifie sa thématique principale (formule libre, courte, 1 à 3 mots).
2. Rédige un résumé en une phrase courte (max 150 caractères).
3. Détermine le sentiment général exprimé : "positif", "négatif" ou "neutre".

Réponds UNIQUEMENT avec du JSON valide, sans aucun texte avant ni après.
Le JSON doit avoir cette structure exacte :
{{
  "analyses": [
    {{
      "verbatim_id": "ID_DU_VERBATIM",
      "theme": "thème en 1-3 mots",
      "summary": "résumé court",
      "sentiment": "positif|négatif|neutre"
    }}
  ]
}}

RÈGLES CRITIQUES :
1. Un objet par verbatim, dans le même ordre que l'entrée.
2. Le verbatim_id doit correspondre exactement à l'ID entre crochets.
3. Le thème doit être court (1 à 3 mots maximum).
4. Le résumé doit être une phrase courte et fidèle au contenu.
5. Le sentiment doit être EXACTEMENT l'une de ces trois valeurs : "positif", "négatif" ou "neutre".
6. Ne jamais inventer d'informations absentes du verbatim.
7. Si le verbatim est dans une langue autre que le français, analyse-le dans sa langue d'origine mais renvoie les champs demandés selon le schéma.

Verbatims :
{lines}
"""


def _extract_json(raw: str) -> Optional[dict]:
    """Extraction robuste du JSON depuis la réponse du LLM.

    Gère les cas courants : markdown fences, texte superflu avant/après.
    """
    # Nettoyer markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Tenter le parsing direct
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback : extraire le premier bloc JSON avec regex
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


class OllamaClient:
    """Client pour l'API Ollama Cloud — async, avec retry."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        url: str = OLLAMA_URL,
        api_key: str = OLLAMA_API_KEY,
        timeout: int = OLLAMA_TIMEOUT,
        temperature: float = TEMPERATURE,
    ):
        self.model = model
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        print(f"🤖 OllamaClient initialisé — modèle: {self.model}")

    def _build_headers(self) -> dict:
        """Construit les headers HTTP avec auth Bearer."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _call_api(self, prompt: str) -> Optional[str]:
        """Appel brut à l'API Ollama — retourne le texte de la réponse."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    json=payload,
                    headers=self._build_headers(),
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")

        except httpx.TimeoutException:
            print(f"  ⏱️  Timeout ({self.timeout}s)")
            return None
        except httpx.HTTPStatusError as e:
            print(f"  ❌ HTTP {e.response.status_code}: {e.response.text[:200]}")
            return None
        except httpx.RequestError as e:
            print(f"  ❌ Erreur réseau: {e}")
            return None

    async def analyze_batch(
        self,
        verbatims: list[VerbatimInput],
        max_retries: int = 2,
    ) -> BatchAnalysisResult:
        """Analyse un batch de verbatims avec retry + backoff exponentiel.

        Raises:
            ValueError: si toutes les tentatives échouent.
        """
        prompt = build_prompt(verbatims)
        last_error = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait = 2 ** attempt  # backoff: 2s, 4s, 8s...
                print(f"  🔄 Retry {attempt}/{max_retries} (attente {wait}s)")
                await asyncio.sleep(wait)

            raw = await self._call_api(prompt)
            if raw is None:
                last_error = "Pas de réponse de l'API"
                continue

            parsed = _extract_json(raw)
            if parsed is None:
                last_error = f"JSON invalide: {raw[:200]}"
                print(f"  ⚠️  {last_error}")
                continue

            try:
                result = BatchAnalysisResult.model_validate(parsed)
                return result
            except Exception as e:
                last_error = f"Validation Pydantic: {e}"
                print(f"  ⚠️  {last_error}")
                continue

        raise ValueError(f"Échec après {max_retries + 1} tentatives: {last_error}")

    async def test_connectivity(self) -> bool:
        """Test rapide de connectivité avec l'API."""
        try:
            raw = await self._call_api("Réponds juste {\"status\": \"ok\"}")
            if raw:
                print(f"  ✅ Connectivité OK — réponse: {raw[:100]}")
                return True
            print("  ❌ Pas de réponse")
            return False
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            return False
