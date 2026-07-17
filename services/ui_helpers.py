"""
Fonctions utilitaires et gestion du design pour l'interface Streamlit.
"""

import os
import pandas as pd
import streamlit as st
from pathlib import Path

# CSS pour un design Premium sombre et contrasté
CUSTOM_CSS = """
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
        color: #94a3b8; /* Gris clair par défaut pour tous les deltas */
    }
    [data-testid="stMetricDelta"] svg {
        fill: currentColor;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
</style>
"""


def inject_custom_css():
    """Injecte le style CSS personnalisé dans l'application Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


def init_env_keys():
    """Charge ou initialise les variables d'environnement dans st.session_state."""
    if "OLLAMA_URL" not in st.session_state:
        st.session_state["OLLAMA_URL"] = os.getenv("OLLAMA_URL", "https://ollama.com/api/chat")
    if "OLLAMA_API_KEY" not in st.session_state:
        st.session_state["OLLAMA_API_KEY"] = os.getenv("OLLAMA_API_KEY", "")
    if "OLLAMA_MODEL" not in st.session_state:
        st.session_state["OLLAMA_MODEL"] = os.getenv("OLLAMA_MODEL", "gemma4:cloud")


def save_env_keys(url, api_key, model):
    """Enregistre les clés d'API configurées dans le .env et en mémoire."""
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
