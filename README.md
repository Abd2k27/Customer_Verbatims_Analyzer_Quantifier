# 🤖 Analyseur et Quantificateur de Verbatims Clients

Ce projet est un pipeline complet, performant et modulaire pour ingérer des commentaires clients (verbatims), en extraire des informations structurées (thématique principale, résumé, sentiment), harmoniser sémantiquement les thèmes libres, et en extraire des taux statistiques quantifiés.

Le pipeline fonctionne de manière asynchrone et effectue des appels à un modèle **Gemma 4** distant hébergé sur **Ollama Cloud** (via authentification Bearer token).

---

## 📁 Architecture du Projet

```
Customer_Verbatims_Analyzer_Quantifier/
├── .env                           # Fichier de configuration contenant les tokens et les clés (gitignored)
├── .env.example                   # Modèle de variables d'environnement
├── .gitignore                     # Protection des secrets et des fichiers générés
├── requirements.txt               # Dépendances Python (pandas, pydantic, httpx, streamlit, plotly)
├── Makefile                       # Automatisation des commandes courantes (run, test, clean, etc.)
├── app.py                         # Application web Streamlit (interface graphique interactive)
│
├── config/
│   ├── __init__.py
│   └── settings.py                # Chargement du .env, constantes et création automatique des dossiers
│
├── models/
│   ├── __init__.py
│   └── schemas.py                 # Modèles de données Pydantic (VerbatimInput, VerbatimAnalysis)
│
├── services/
│   ├── __init__.py
│   ├── llm_client.py              # Client HTTP Ollama Cloud (retry, backoff, parsing JSON robuste)
│   ├── validator.py               # Validation métier (taille de thèmes, IDs dupliqués, etc.)
│   └── harmonizer.py              # Module LLM de regroupement des thèmes libres uniques
│
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py               # Chargement et normalisation des sources (CSV, JSONL, DataFrame)
│   ├── orchestrator.py            # Orchestrateur de batchs avec checkpoints et reprise sur erreur
│   └── persistence.py             # Export incrémental des résultats (.jsonl) et erreurs (.jsonl)
│
├── scripts/
│   ├── download_datasets.py       # Téléchargement et préparation de datasets de test depuis HuggingFace
│   ├── run_pipeline.py            # Point d'entrée principal en ligne de commande (CLI)
│   └── quantify.py                # Script CLI d'harmonisation et de quantification
│   └── generate_multilingual_dataset.py # Générateur de dataset de test multilingue [NOUVEAU]
│
└── data/                          # Dossier de stockage des données (créé automatiquement)
    ├── raw/                       # Fichiers sources bruts (.csv)
    ├── processed/                 # Analyses générées (.jsonl) et rapports statistiques (.json)
    └── checkpoints/               # État d'avancement des runs pour la reprise idempotente
```

---

## ⚙️ Explication par Fonctionnalité et Fonctions Codées

### 1. Ingestion et Normalisation (`pipeline/ingestion.py`)
Ce module convertit n'importe quel fichier de commentaires en une liste standardisée d'objets `VerbatimInput` avec détection automatique de colonnes (texte, ID, source, langue).

*   `_detect_column(df, candidates)`: Recherche s'il existe une correspondance insensible à la casse entre les colonnes du fichier et des candidats typiques (ex: `review`, `comment` pour le texte; `source`, `platform` pour la source).
*   `load_csv(path, text_col, id_col, source, lang, limit)`: Charge un fichier CSV, applique l'auto-détection pour toutes les colonnes clés (y compris `source` et `lang` si présentes dans le CSV), et retourne une liste de `VerbatimInput`.
*   `load_jsonl(...)`: Même logique que pour le CSV, mais pour les fichiers JSON Lines.
*   `from_dataframe(...)`: Convertit directement un DataFrame pandas chargé en mémoire (par exemple via Streamlit) en liste de `VerbatimInput` en extrayant dynamiquement le texte, l'ID, la source et la langue.

---

### 2. Client Ollama Cloud (`services/llm_client.py`)
Ce service gère la communication HTTP asynchrone avec l'API Ollama Cloud, le formatage des requêtes et le parsing robuste.

*   `build_prompt(verbatims)`: Génère le prompt pour le modèle. Il inclut la liste des verbatims à traiter (avec leurs identifiants) et dicte un schéma JSON strict contenant les clés `verbatim_id`, `theme`, `summary` et `sentiment`.
*   `_extract_json(raw_text)`: Extrait de manière robuste le JSON de la réponse. Supprime les délimiteurs markdown (fences ```json ... ```) et utilise une expression régulière (`\{.*\}`) pour extraire le dictionnaire JSON en cas de bavardage superflu du LLM.
*   `OllamaClient._call_api(prompt)`: Effectue l'appel POST asynchrone à l'endpoint avec header `Authorization: Bearer <api_key>` en limitant la température à `0.2` (pour limiter la dérive créative des thèmes).
*   `OllamaClient.analyze_batch(verbatims, max_retries)`: Envoie les données au modèle. Si l'appel ou la validation Pydantic échoue, elle réessaie automatiquement en appliquant un **backoff exponentiel** (délai doublé à chaque tentative : 2s, 4s, 8s...).

---

### 3. Validation Métier (`services/validator.py`)
Cette couche vérifie les résultats retournés par le modèle pour s'assurer de l'intégrité de la donnée avant la sauvegarde.

*   `validate_batch_result(result, expected_verbatims)`:
    -   Vérifie que le nombre d'analyses retournées correspond au nombre de verbatims envoyés (`count_mismatch`).
    -   Vérifie que chaque verbatim_id retourné existe dans le batch d'origine (`id_inconnu`).
    -   Détecte les doublons d'IDs.
    -   Valide la longueur du thème (rejet si le thème contient plus de 5 mots).
    -   Valide la longueur minimale du résumé.
    -   Isole les éléments valides et génère des logs d'erreurs précis pour les éléments invalides ou sautés par le LLM.

---

### 4. Orchestrateur et Reprise sur Erreur (`pipeline/orchestrator.py`)
L'orchestrateur découpe les données, suit la progression sur le disque et gère l'idempotence du run.

*   `_load_checkpoint()` / `_save_checkpoint(...)`: Charge et sauvegarde un fichier `progress.json` dans le dossier de checkpoint du run. Ce fichier enregistre la liste des index de lots (batches) traités avec succès.
*   `PipelineOrchestrator.run(verbatims, output_path, resume, on_progress)`:
    -   Découpe la liste en lots de taille `batch_size`.
    -   Si `resume=True`, l'orchestrateur regarde le checkpoint et saute les lots déjà analysés.
    -   Pour chaque lot, il appelle le client LLM et le validateur.
    -   **Enrichissement à la volée** : Il injecte directement le texte d'origine (`texte_original`) et la source d'origine (`source`) de chaque verbatim dans les résultats d'analyses validés avant de les enregistrer (évite les jointures de fichiers distants).
    -   Écrit ces résultats enrichis en mode *append* sur le disque et met à jour le checkpoint.
    -   En cas d'échec persistant d'un lot (retries épuisés), il logue l'échec et passe au lot suivant **sans** le marquer comme complété pour permettre de le rejouer plus tard.
    -   Déclenche un callback `on_progress` après chaque lot pour notifier l'interface graphique Streamlit.

---

### 5. Harmonisation Sémantique (`services/harmonizer.py`)
Pour agréger et quantifier proprement, cette classe réduit la liste des thèmes libres uniques en un nombre restreint de thèmes harmonisés.

*   `build_harmonization_prompt(unique_themes, max_categories)`: Demande au LLM de concevoir jusqu'à `N` macro-catégories à partir d'une liste brute de thèmes uniques et de renvoyer un mapping structuré rédigé obligatoirement en français.
*   `ThemeHarmonizer.harmonize(unique_themes, max_categories)`: Envoie les thèmes bruts au modèle, valide la réponse sous forme de dictionnaire de mapping `{thème_libre: macro_catégorie}`, et gère un fallback automatique vers une catégorie "Autre" pour les thèmes oubliés.

---

### 6. Synthèse Qualitative (`services/synthesizer.py`) [NOUVEAU]
Ce service produit des synthèses sémantiques résumant les forces et les faiblesses des retours clients pour chaque grande thématique macro.

*   `build_synthesis_prompt(macro_theme, reviews)`: Génère le prompt LLM en lui passant la liste compacte des couples `[sentiment] résumé` pour la catégorie concernée afin d'être optimal en termes de coûts/tokens. Il lui dicte de produire un JSON contenant les points forts (`positive_points`), les irritants (`negative_points`) et un paragraphe de synthèse (`global_synthesis`) rédigé en français.
*   `CategorySynthesizer.synthesize(macro_theme, analyses)`: Appelle l'API Ollama Cloud de façon asynchrone, extrait la réponse et valide l'objet final sous forme de `CategorySynthesis`.

---

### 7. Quantification et Export (`scripts/quantify.py`)
Calcul des statistiques de sentiment et mise en forme des rapports.

*   `display_report(...)`: Affiche un rapport complet dans le terminal sous forme de graphiques en barres horizontales.
*   `main()`: Charge un fichier de résultats `.jsonl`, extrait les thèmes, appelle l'harmonisation, calcule la distribution des sentiments globaux et par thématique macro, puis exporte le fichier de résultats enrichi et le fichier de statistiques `.json`.

---

### 8. Interface Web Streamlit (`app.py`)
L'application unifie l'ensemble des modules dans un tableau de bord réactif.

*   `load_all_raw_metadata()` : Scanne localement le dossier `data/raw/` pour y charger le texte et la source d'origine de chaque verbatim en cas de rétrocompatibilité avec d'anciens fichiers d'analyse.
*   **Affichage direct du texte et de la source** : L'interface extrait directement le texte et la source stockés dans le fichier de résultats `.jsonl`.
*   **Logique de filtrage dynamique** : Permet de filtrer l'ensemble du dashboard (KPIs métriques, graphique Donut Plotly de sentiment global, graphique barres Plotly cumulées) par **Source** en temps réel.
*   **Génération interactive des synthèses** : Propose un bouton à la demande pour synthétiser les avis par catégorie. Les résultats générés (points forts, irritants, résumé global) sont intégrés dans le rapport JSON pour ne pas avoir à les recalculer.
*   **Visualiseur filtrable multi-critères** : Tableaux Streamlit avec recherche textuelle intégrée et filtres par Source, Thème Macro, et Sentiment.
*   **Masquage de sécurité** : Masque le token d'API Ollama Cloud dans l'interface et utilise de manière sécurisée un placeholder si ce dernier est déjà présent dans les variables système ou secrets de Streamlit Cloud.

---

## 🚀 Démarrage Rapide

### 1. Installation
Installez les dépendances :
```bash
make install
```

### 2. Configuration (.env)
Renseignez votre clé d'API et l'endpoint dans le fichier `.env` :
```env
OLLAMA_URL=https://ollama.com/api/chat
OLLAMA_API_KEY=votre_clé_api
OLLAMA_MODEL=gemma4:cloud
```

### 3. Télécharger les datasets de test
```bash
make download DATASET=all LIMIT=200
```

### 4. Lancer l'interface web
```bash
make streamlit
```
L'interface s'ouvre alors dans votre navigateur (par défaut sur `http://localhost:8501`).
