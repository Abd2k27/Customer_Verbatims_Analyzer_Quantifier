"""
Application Streamlit — Analyse de sentiment et quantification de verbatims clients.

Interface graphique moderne et interactive pour charger des verbatims,
lancer l'analyse LLM via Ollama Cloud, harmoniser les thématiques et
visualiser les statistiques de sentiment.
"""

import sys
import os
import asyncio
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from datetime import datetime

# Ajouter le root du projet au path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import RAW_DIR, PROCESSED_DIR
from models.schemas import VerbatimAnalysis
from pipeline.ingestion import from_dataframe
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.persistence import load_results
from services.llm_client import OllamaClient
from services.harmonizer import ThemeHarmonizer
from services.synthesizer import CategorySynthesizer

# Helper pour charger les métadonnées des verbatims (texte et source)
def load_all_raw_metadata():
    """Scanne data/raw/ pour charger la source et le texte originaux par verbatim_id."""
    mapping = {}
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return mapping
    
    for csv_file in raw_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            id_col = None
            for cand in ["id", "review_id", "verbatim_id", "index"]:
                if cand in df.columns:
                    id_col = cand
                    break
            
            # Si le CSV n'a pas de colonne source, on utilise le nom du fichier par défaut
            source_col = "source" if "source" in df.columns else None
            text_col = None
            for cand in ["text", "review", "review_body", "review_text", "comment", "verbatim", "content"]:
                if cand in df.columns:
                    text_col = cand
                    break
            
            if text_col:
                for idx, row in df.iterrows():
                    vid = str(row[id_col]) if id_col else f"{csv_file.stem.split('_')[0]}_{idx}"
                    src = str(row[source_col]) if source_col else csv_file.stem.split('_')[0]
                    mapping[vid] = {
                        "text": str(row[text_col]),
                        "source": src
                    }
        except Exception:
            continue
    return mapping


# Page configuration
st.set_page_config(
    page_title="Analyste de Sentiment Clients",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .stCard {
        background-color: #1e222b;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    /* Amélioration de la lisibilité des cartes d'indicateurs (st.metric) */
    [data-testid="stMetric"] {
        background-color: #1e222b !important;
        border: 1px solid #3f4452 !important;
        border-radius: 10px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15) !important;
    }
    /* Titre (Label) de l'indicateur */
    [data-testid="stMetricLabel"] p {
        color: #cbd5e1 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    /* Valeur principale de l'indicateur */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }
    /* Texte secondaire du delta */
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        color: #94a3b8; /* Gris clair par défaut pour tous les deltas (laissé surchargeable par vert/rouge) */
    }
    [data-testid="stMetricDelta"] svg {
        fill: currentColor;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


def init_env_keys():
    """Charge ou initialise les variables du .env dans st.session_state."""
    if "OLLAMA_URL" not in st.session_state:
        st.session_state["OLLAMA_URL"] = os.getenv("OLLAMA_URL", "https://ollama.com/api/chat")
    if "OLLAMA_API_KEY" not in st.session_state:
        st.session_state["OLLAMA_API_KEY"] = os.getenv("OLLAMA_API_KEY", "")
    if "OLLAMA_MODEL" not in st.session_state:
        st.session_state["OLLAMA_MODEL"] = os.getenv("OLLAMA_MODEL", "gemma4:cloud")


def save_env_keys(url, api_key, model):
    """Enregistre les variables dans le fichier .env et en mémoire."""
    st.session_state["OLLAMA_URL"] = url
    st.session_state["OLLAMA_API_KEY"] = api_key
    st.session_state["OLLAMA_MODEL"] = model

    # Réécriture propre du .env
    env_lines = [
        f"OLLAMA_URL={url}",
        f"OLLAMA_API_KEY={api_key}",
        f"OLLAMA_MODEL={model}",
        "OLLAMA_TIMEOUT=120",
        "BATCH_SIZE=10",
        "MAX_RETRIES=2",
        "TEMPERATURE=0.2"
    ]
    with open(".env", "w") as f:
        f.write("\n".join(env_lines) + "\n")

    # Mettre à jour l'environnement OS courant
    os.environ["OLLAMA_URL"] = url
    os.environ["OLLAMA_API_KEY"] = api_key
    os.environ["OLLAMA_MODEL"] = model


# Initialiser la config
init_env_keys()

# === SIDEBAR ===
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    # Ollama Credentials
    st.subheader("Ollama Cloud API")
    env_url = st.text_input("URL Ollama Endpoint", value=st.session_state["OLLAMA_URL"])
    
    # Sécurisation de l'affichage de l'API key
    has_key_in_env = bool(os.getenv("OLLAMA_API_KEY")) or bool(st.session_state.get("OLLAMA_API_KEY"))
    placeholder_text = "🔒 Déjà configurée en tâche de fond (Secrets/Env)" if has_key_in_env else "Saisir la clé d'API..."
    
    env_key_input = st.text_input(
        "Ollama API Key / Bearer Token",
        value="", 
        type="password",
        placeholder=placeholder_text,
        help="Laissez vide pour utiliser la clé déjà configurée dans vos secrets Streamlit ou le fichier .env."
    )
    
    # Utiliser la clé saisie par l'utilisateur si présente, sinon celle en mémoire/env
    env_key = env_key_input if env_key_input else st.session_state.get("OLLAMA_API_KEY", "")
    
    env_model = st.text_input("Modèle", value=st.session_state["OLLAMA_MODEL"])

    if st.button("Sauvegarder Configuration", width="stretch"):
        final_key = env_key_input if env_key_input else st.session_state.get("OLLAMA_API_KEY", "")
        save_env_keys(env_url, final_key, env_model)
        st.success("Configuration enregistrée !")

    st.markdown("---")

    # Hyperparamètres Pipeline
    st.subheader("Paramètres de Run")
    batch_size = st.slider("Taille du Batch", min_value=1, max_value=20, value=10)
    limit = st.number_input("Limite de verbatims (0 = tout charger)", min_value=0, value=100)
    max_retries = st.slider("Nombre de Retries", min_value=0, max_value=5, value=2)

    # Test connectivité
    if st.button("🔌 Tester Connexion Cloud", width="stretch"):
        client = OllamaClient(model=env_model, url=env_url, api_key=env_key)
        with st.spinner("Test en cours..."):
            ok = asyncio.run(client.test_connectivity())
            if ok:
                st.success("Connexion réussie ! L'API Ollama Cloud répond.")
            else:
                st.error("Échec de connexion. Vérifie l'URL et la clé d'API.")

# === MAIN INTERFACE ===
st.title("🤖 Analyseur et Quantificateur de Verbatims Clients")
st.markdown("Prenez des verbatims bruts, extrayez le thème, le résumé et le sentiment, puis regroupez-les sémantiquement.")

tabs = st.tabs(["🚀 Lancer l'Analyse", "📊 Dashboard & Quantification"])

# TAB 1: RUN PIPELINE
with tabs[0]:
    st.header("Charger des Verbatims")

    uploaded_file = st.file_uploader(
        "Déposer un fichier CSV ou JSONL contenant les commentaires clients",
        type=["csv", "jsonl"]
    )

    if uploaded_file is not None:
        # Lire le fichier
        if uploaded_file.name.endswith(".csv"):
            df_preview = pd.read_csv(uploaded_file)
        else:
            df_preview = pd.read_json(uploaded_file, lines=True)

        st.subheader("Aperçu des données")
        st.dataframe(df_preview, width="stretch")

        # Sélection des colonnes
        cols = list(df_preview.columns)

        # Auto-détection
        default_text_col = None
        for cand in ["text", "review", "review_body", "review_text", "comment", "verbatim", "content"]:
            if cand in cols:
                default_text_col = cand
                break

        default_id_col = None
        for cand in ["id", "review_id", "verbatim_id", "index"]:
            if cand in cols:
                default_id_col = cand
                break

        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox(
                "Sélectionnez la colonne contenant le TEXTE",
                options=cols,
                index=cols.index(default_text_col) if default_text_col else 0
            )
        with col2:
            id_col = st.selectbox(
                "Sélectionnez la colonne contenant l'ID (optionnel)",
                options=["(Générer des IDs automatiquement)"] + cols,
                index=cols.index(default_id_col) + 1 if default_id_col else 0
            )

        lang = st.selectbox("Langue par défaut des verbatims", options=["fr", "en", "es", "it", "de"], index=0)

        # Préparer les inputs
        limit_val = None if limit == 0 else int(limit)
        id_col_val = None if id_col == "(Générer des IDs automatiquement)" else id_col

        verbatims_input = from_dataframe(
            df_preview,
            text_col=text_col,
            id_col=id_col_val,
            source=uploaded_file.name.split(".")[0],
            lang=lang,
            limit=limit_val
        )

        st.info(f"📋 {len(verbatims_input)} verbatims prêts à être analysés par batch de {batch_size}.")

        # Bouton RUN
        if st.button("🚀 Démarrer l'Analyse Cloud", type="primary", width="stretch"):
            # Zone d'affichage de la progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_area = st.expander("Logs détaillés du traitement", expanded=True)

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = PROCESSED_DIR / f"results_{run_id}.jsonl"

            # Callback de progression pour Streamlit
            def on_progress(batch_idx, total_batches, completed_count, error_count):
                progress = batch_idx / total_batches
                progress_bar.progress(progress)
                status_text.markdown(
                    f"**Progression** : Batch {batch_idx}/{total_batches} traité "
                    f"({completed_count} analyses valides, {error_count} erreurs)"
                )

            # Lancement asynchrone orchestré
            async def run_async_pipeline():
                client = OllamaClient(model=env_model, url=env_url, api_key=env_key)
                orchestrator = PipelineOrchestrator(
                    client=client,
                    batch_size=batch_size,
                    max_retries=max_retries,
                    run_id=run_id
                )
                return await orchestrator.run(
                    verbatims_input,
                    output_path=output_file,
                    resume=False,
                    on_progress=on_progress
                )

            with st.spinner("Analyse LLM en cours..."):
                try:
                    results, errors = asyncio.run(run_async_pipeline())

                    st.success(f"🎉 Analyse terminée avec succès !")
                    st.balloons()

                    # Résumé rapide
                    st.markdown(f"### Résumé du Run : `{run_id}`")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Analyses Réussies", len(results))
                    c2.metric("Erreurs rencontrées", len(errors))
                    c3.markdown(f"**Fichier généré** :\n`{output_file.name}`")

                    # Basculer vers l'onglet dashboard
                    st.info("💡 Vous pouvez maintenant aller dans l'onglet **Dashboard & Quantification** pour analyser les sentiments et thèmes.")
                except Exception as e:
                    st.error(f"Une erreur est survenue lors de l'exécution : {e}")

# TAB 2: DASHBOARD & QUANTIFICATION
with tabs[1]:
    st.header("Visualisation des résultats et quantification")

    # Lister les fichiers de résultats disponibles
    result_files = sorted(list(PROCESSED_DIR.glob("results_*.jsonl")), reverse=True)

    if not result_files:
        st.warning("Aucun fichier de résultats trouvé. Veuillez d'abord lancer une analyse dans l'onglet précédent.")
    else:
        file_options = {f.name: f for f in result_files}
        selected_filename = st.selectbox(
            "Sélectionner un fichier de résultats à analyser",
            options=list(file_options.keys())
        )
        selected_file_path = file_options[selected_filename]

        # Charger les résultats
        raw_analyses = load_results(selected_file_path)
        total_verbatims = len(raw_analyses)

        st.subheader(f"Données brutes extraites — {total_verbatims} commentaires")

        # Convertir en DataFrame pour l'UI
        df_results = pd.DataFrame([r.model_dump() for r in raw_analyses])
        
        # Charger et mapper le texte original et la source (avec fallback s'ils ne sont pas déjà présents dans le fichier de résultats)
        raw_meta = load_all_raw_metadata()
        if not df_results.empty:
            # Remplir le texte original si manquant ou vide
            if "texte_original" not in df_results.columns or df_results["texte_original"].isna().all() or (df_results["texte_original"] == "").all():
                df_results["texte_original"] = df_results["verbatim_id"].apply(lambda vid: raw_meta.get(vid, {}).get("text", "(Texte original introuvable)"))
            else:
                df_results["texte_original"] = df_results.apply(
                    lambda row: row["texte_original"] if pd.notna(row["texte_original"]) and row["texte_original"] != ""
                    else raw_meta.get(row["verbatim_id"], {}).get("text", "(Texte original introuvable)"),
                    axis=1
                )
            
            # Remplir la source si manquante ou inconnue
            if "source" not in df_results.columns or df_results["source"].isna().all() or (df_results["source"] == "inconnue").all():
                df_results["source"] = df_results["verbatim_id"].apply(lambda vid: raw_meta.get(vid, {}).get("source", "inconnue"))
            else:
                df_results["source"] = df_results.apply(
                    lambda row: row["source"] if pd.notna(row["source"]) and row["source"] != "inconnue"
                    else raw_meta.get(row["verbatim_id"], {}).get("source", "inconnue"),
                    axis=1
                )

            # Réordonner
            cols_order = ["verbatim_id", "source", "texte_original", "theme", "summary", "sentiment"]
            if "macro_theme" in df_results.columns:
                cols_order.insert(3, "macro_theme")
            cols_order = [c for c in cols_order if c in df_results.columns]
            df_results = df_results[cols_order]

        st.dataframe(df_results, width="stretch")

        st.markdown("---")

        # Section Harmonisation des Thématiques
        st.header("🧠 Étape 2 : Harmonisation Sémantique des Thèmes libres")
        st.markdown("Les thèmes extraits librement par le LLM peuvent être fragmentés sémantiquement. Nous allons demander au LLM de les catégoriser en macro-thématiques uniques.")

        max_cats = st.slider("Nombre maximal de macro-catégories à créer", min_value=3, max_value=10, value=5)

        # Vérifier si un rapport d'harmonisation existe déjà pour ce fichier
        run_id_from_file = selected_filename.replace("results_", "").replace(".jsonl", "")
        report_file_path = PROCESSED_DIR / f"quantification_report_{run_id_from_file}.json"
        enriched_file_path = PROCESSED_DIR / f"enriched_results_{run_id_from_file}.jsonl"

        # Bouton d'harmonisation
        harmonize_clicked = st.button("🧠 Regrouper et Harmoniser les Thèmes", type="primary")

        if harmonize_clicked or report_file_path.exists():
            report_data = None

            if harmonize_clicked:
                with st.spinner("Classification sémantique en cours par le LLM..."):
                    # Extraire les thèmes uniques
                    unique_themes = sorted(list(df_results["theme"].unique()))

                    async def run_harmonization():
                        client = OllamaClient(model=env_model, url=env_url, api_key=env_key)
                        harmonizer = ThemeHarmonizer(client=client)
                        return await harmonizer.harmonize(unique_themes, max_categories=max_cats)

                    try:
                        harmonization = asyncio.run(run_harmonization())

                        # Appliquer le mapping
                        df_results["macro_theme"] = df_results["theme"].map(harmonization.mapping).fillna("Autre")

                        # Re-calculer les stats
                        global_sentiment = df_results["sentiment"].value_counts()
                        global_sentiment_dict = {k: int(v) for k, v in global_sentiment.items()}
                        # Sécuriser les clés manquantes
                        for k in ["positif", "négatif", "neutre"]:
                            global_sentiment_dict.setdefault(k, 0)

                        # Distribution par thématique macro
                        theme_distribution = {}
                        for cat in harmonization.categories:
                            theme_distribution[cat] = {"total": 0, "positif": 0, "négatif": 0, "neutre": 0}

                        for _, row in df_results.iterrows():
                            m_theme = row["macro_theme"]
                            sent = row["sentiment"]
                            if m_theme not in theme_distribution:
                                theme_distribution[m_theme] = {"total": 0, "positif": 0, "négatif": 0, "neutre": 0}
                            theme_distribution[m_theme]["total"] += 1
                            theme_distribution[m_theme][sent] += 1

                        # Structurer le rapport
                        report_data = {
                            "metadata": {
                                "source_file": str(selected_file_path),
                                "timestamp": datetime.now().isoformat(),
                                "total_verbatims": total_verbatims,
                            },
                            "global_sentiment": global_sentiment_dict,
                            "theme_distribution": theme_distribution,
                            "mapping": harmonization.mapping
                        }

                        # Sauvegarder sur le disque
                        with open(report_file_path, "w", encoding="utf-8") as f:
                            json.dump(report_data, f, indent=2, ensure_ascii=False)

                        # Sauvegarder les résultats enrichis
                        df_results.to_json(enriched_file_path, orient="records", lines=True, force_ascii=False)

                        st.success("Harmonisation terminée !")

                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de l'harmonisation : {e}")

            # Recharger le rapport depuis le fichier s'il existait déjà
            if report_data is None and report_file_path.exists():
                with open(report_file_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)

                # Mettre à jour df_results avec le macro_theme enrichi si le fichier existe
                if enriched_file_path.exists():
                    df_results = pd.read_json(enriched_file_path, lines=True)

            if report_data:
                # S'assurer que les métadonnées sont chargées pour df_results
                if "source" not in df_results.columns:
                    df_results["source"] = df_results["verbatim_id"].apply(lambda vid: raw_meta.get(vid, {}).get("source", "inconnue"))
                if "texte_original" not in df_results.columns:
                    df_results["texte_original"] = df_results["verbatim_id"].apply(lambda vid: raw_meta.get(vid, {}).get("text", "(Texte original introuvable)"))

                st.markdown("---")
                st.header("📊 DASHBOARD STATISTIQUE & VISUALISATIONS")

                # Sélecteur de source interactif
                available_sources = sorted(list(df_results["source"].unique()))
                selected_sources = st.multiselect(
                    "Filtrer le Dashboard par Source",
                    options=available_sources,
                    default=[],
                    help="Laissez vide pour afficher toutes les sources"
                )

                # Filtrer les données pour les calculs du Dashboard
                df_dashboard = df_results.copy()
                if selected_sources:
                    df_dashboard = df_dashboard[df_dashboard["source"].isin(selected_sources)]

                # 1. indicateurs de synthèse (Kpi) dynamiques
                st.subheader("Indicateurs Clés")
                col_tot, col_pos, col_neg, col_neu = st.columns(4)

                tot = len(df_dashboard)
                if tot > 0:
                    pos = len(df_dashboard[df_dashboard["sentiment"] == "positif"])
                    neg = len(df_dashboard[df_dashboard["sentiment"] == "négatif"])
                    neu = len(df_dashboard[df_dashboard["sentiment"] == "neutre"])

                    col_tot.metric("Commentaires Analysés", tot)
                    col_pos.metric("Sentiment Positif", f"{(pos / tot) * 100:.1f}%", f"{pos} verbatims")
                    col_neg.metric("Sentiment Négatif", f"{(neg / tot) * 100:.1f}%", f"-{neg} verbatims", delta_color="inverse")
                    col_neu.metric("Sentiment Neutre", f"{(neu / tot) * 100:.1f}%", f"{neu} verbatims", delta_color="off")

                    # 2. Graphiques
                    st.markdown("### Visualisations Graphiques")
                    g1, g2 = st.columns([1, 2])

                    with g1:
                        st.markdown("#### Répartition Globale")
                        df_sent_pie = pd.DataFrame({
                            "Sentiment": ["positif", "négatif", "neutre"],
                            "Nombre": [pos, neg, neu]
                        })
                        fig_pie = px.pie(
                            df_sent_pie,
                            values="Nombre",
                            names="Sentiment",
                            color="Sentiment",
                            color_discrete_map={"positif": "#2ecc71", "négatif": "#e74c3c", "neutre": "#95a5a6"},
                            hole=0.4
                        )
                        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, width="stretch")

                    with g2:
                        st.markdown("#### Taux de Sentiment par Macro-Thématique")

                        if "macro_theme" in df_dashboard.columns:
                            # Calculer la distribution groupée
                            df_theme_bars = df_dashboard.groupby(["macro_theme", "sentiment"]).size().reset_index(name="Nombre")
                            df_theme_bars.rename(columns={"macro_theme": "Catégorie"}, inplace=True)

                            if not df_theme_bars.empty:
                                fig_bar = px.bar(
                                    df_theme_bars,
                                    x="Nombre",
                                    y="Catégorie",
                                    color="sentiment",
                                    orientation="h",
                                    color_discrete_map={"positif": "#2ecc71", "négatif": "#e74c3c", "neutre": "#95a5a6"},
                                    barmode="stack"
                                )
                                fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                                st.plotly_chart(fig_bar, width="stretch")
                            else:
                                st.info("Pas assez de données pour afficher le graphique par thématique.")
                        else:
                            st.info("Catégories macro non disponibles.")
                else:
                    st.warning("Aucune donnée disponible pour les filtres sélectionnés.")

                # 3. Synthèse Qualitative par Macro-Catégorie
                st.markdown("---")
                st.header("🧠 SYNTHÈSE QUALITATIVE PAR MACRO-CATÉGORIE")
                st.markdown("Analyse automatique par LLM des forces et irritants pour chaque service.")

                # Initialiser le dictionnaire des synthèses si absent
                if "syntheses" not in report_data:
                    report_data["syntheses"] = {}

                # Trouver les catégories disponibles
                macro_themes_list = sorted(list(report_data["theme_distribution"].keys()))

                # Proposer de générer ou régénérer les synthèses
                gen_col1, gen_col2 = st.columns([3, 1])
                with gen_col1:
                    st.info("La génération de la synthèse qualitative par le LLM prend environ 10 à 20 secondes.")
                with gen_col2:
                    generate_synthesis = st.button("🧠 Générer la Synthèse", type="primary", key="btn_gen_synth", width="stretch")

                if generate_synthesis:
                    syntheses_temp = {}
                    progress_bar_synth = st.progress(0)
                    status_synth = st.empty()
                    
                    # Filtrer les analyses valides de ce run_id
                    client = OllamaClient(model=env_model, url=env_url, api_key=env_key)
                    synthesizer = CategorySynthesizer(client=client)
                    
                    for idx, cat in enumerate(macro_themes_list):
                        status_synth.write(f"Analyse en cours pour la catégorie **{cat}**...")
                        # Filtrer les analyses correspondantes
                        cat_analyses = [
                            VerbatimAnalysis(
                                verbatim_id=row["verbatim_id"],
                                theme=row["theme"],
                                summary=row["summary"],
                                sentiment=row["sentiment"]
                            )
                            for _, row in df_results[df_results["macro_theme"] == cat].iterrows()
                        ]
                        
                        try:
                            synth = asyncio.run(synthesizer.synthesize(cat, cat_analyses))
                            syntheses_temp[cat] = synth.model_dump()
                        except Exception as e:
                            st.error(f"Erreur lors de la synthèse de {cat} : {e}")
                            
                        progress_bar_synth.progress((idx + 1) / len(macro_themes_list))
                    
                    status_synth.empty()
                    progress_bar_synth.empty()
                    
                    # Enregistrer dans le rapport
                    report_data["syntheses"] = syntheses_temp
                    with open(report_file_path, "w", encoding="utf-8") as f:
                        json.dump(report_data, f, indent=2, ensure_ascii=False)
                    st.success("Synthèse qualitative générée avec succès !")
                    st.rerun()

                # Afficher les synthèses si elles existent
                if report_data.get("syntheses"):
                    for cat in macro_themes_list:
                        synth_item = report_data["syntheses"].get(cat)
                        if synth_item:
                            with st.expander(f"🔹 {cat} — Synthèse qualitative", expanded=True):
                                st.markdown(f"**Synthèse globale :** *{synth_item.get('global_synthesis')}*")
                                c_pos, c_neg = st.columns(2)
                                with c_pos:
                                    st.markdown("##### 🟢 Points de Satisfaction / Forces")
                                    pos_list = synth_item.get("positive_points", [])
                                    if pos_list:
                                        for pt in pos_list:
                                            st.markdown(f"- {pt}")
                                    else:
                                        st.markdown("*Aucun point fort récurrent détecté.*")
                                with c_neg:
                                    st.markdown("##### 🔴 Points d'Insatisfaction / Irritants")
                                    neg_list = synth_item.get("negative_points", [])
                                    if neg_list:
                                        for pt in neg_list:
                                            st.markdown(f"- {pt}")
                                    else:
                                        st.markdown("*Aucun irritant majeur détecté.*")
                else:
                    st.warning("Aucune synthèse générée pour le moment. Veuillez cliquer sur le bouton ci-dessus pour la lancer.")

                # 3. Explorateur filtrable
                st.markdown("### 🔎 Explorateur de Verbatims")

                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    filter_source = st.multiselect(
                        "Filtrer l'explorateur par Source",
                        options=available_sources
                    )
                with col_f2:
                    filter_theme = st.multiselect(
                        "Filtrer l'explorateur par Catégorie Macro",
                        options=list(report_data["theme_distribution"].keys())
                    )
                with col_f3:
                    filter_sentiment = st.multiselect(
                        "Filtrer l'explorateur par Sentiment",
                        options=["positif", "négatif", "neutre"]
                    )

                # Filtrage pandas
                df_filtered = df_results.copy()
                if filter_source:
                    df_filtered = df_filtered[df_filtered["source"].isin(filter_source)]
                if filter_theme:
                    df_filtered = df_filtered[df_filtered["macro_theme"].isin(filter_theme)]
                if filter_sentiment:
                    df_filtered = df_filtered[df_filtered["sentiment"].isin(filter_sentiment)]

                # S'assurer que la colonne texte_original et la source sont bien présentes dans le dataframe filtré
                cols_to_show = ["verbatim_id", "source", "texte_original", "macro_theme", "theme", "sentiment", "summary"]
                cols_to_show = [c for c in cols_to_show if c in df_filtered.columns]
                st.dataframe(df_filtered[cols_to_show], width="stretch")

                # 4. Exports et téléchargements
                st.markdown("### 💾 Téléchargements des rapports")
                d1, d2 = st.columns(2)

                with d1:
                    if enriched_file_path.exists():
                        with open(enriched_file_path, "r", encoding="utf-8") as f:
                            st.download_button(
                                label="📥 Télécharger les Résultats Enrichis (JSONL)",
                                data=f.read(),
                                file_name=f"enriched_results_{run_id_from_file}.jsonl",
                                mime="application/jsonlines",
                                width="stretch"
                            )

                with d2:
                    if report_file_path.exists():
                        with open(report_file_path, "r", encoding="utf-8") as f:
                            st.download_button(
                                label="📥 Télécharger le Rapport Statistique (JSON)",
                                data=f.read(),
                                file_name=f"quantification_report_{run_id_from_file}.json",
                                mime="application/json",
                                width="stretch"
                            )
