# Configuration et constantes du projet
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()
# Charger aussi depuis le dossier parent si présent
parent_env = Path(__file__).parent.parent / ".env"
if parent_env.exists():
    load_dotenv(parent_env, override=True)

# === Chemins du projet ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# Créer les répertoires s'ils n'existent pas
for d in [RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Configuration Ollama Cloud ===
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:cloud")
OLLAMA_URL = os.getenv("OLLAMA_URL", "https://ollama.com/api/chat")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# === Configuration Pipeline ===
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
