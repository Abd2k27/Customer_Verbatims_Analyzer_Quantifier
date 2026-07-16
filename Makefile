.PHONY: help test run clean clean-cache clean-checkpoints clean-results clean-all download quantify streamlit generate-multilingual

# === Configuration ===
PYTHON := python3
INPUT ?= data/raw/test_sample.csv
LIMIT ?= 100
BATCH_SIZE ?= 10
DATASET ?= test

help: ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# === Exécution ===
test: ## Test de connectivité Ollama Cloud
	$(PYTHON) scripts/run_pipeline.py --test

run: ## Lance le pipeline (INPUT=... LIMIT=... BATCH_SIZE=...)
	$(PYTHON) scripts/run_pipeline.py -i $(INPUT) -l $(LIMIT) -b $(BATCH_SIZE)

run-test: ## Lance sur le fichier test (8 verbatims)
	$(PYTHON) scripts/run_pipeline.py -i data/raw/test_sample.csv -b 4

quantify: ## Lance l'harmonisation et la quantification (INPUT=...)
	$(PYTHON) scripts/quantify.py -i $(INPUT)

streamlit: ## Lance l'interface web Streamlit
	streamlit run app.py

generate-multilingual: ## Génère un dataset de test multilingue de 500 verbatims (EN, FR, ES, PT, DE, BE)
	$(PYTHON) scripts/generate_multilingual_dataset.py

download: ## Télécharge les datasets (DATASET=test|allocine|bank|amazon|all LIMIT=...)
	$(PYTHON) scripts/download_datasets.py --dataset $(DATASET) --limit $(LIMIT)

# === Nettoyage ===
clean-cache: ## Supprime les __pycache__ et .pyc
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

clean-checkpoints: ## Supprime les checkpoints (data/checkpoints/)
	rm -rf data/checkpoints/*
	@echo "✅ Checkpoints supprimés"

clean-results: ## Supprime les résultats générés (data/processed/)
	rm -rf data/processed/*
	@echo "✅ Résultats supprimés"

clean-data: ## Supprime les données brutes téléchargées (data/raw/)
	rm -rf data/raw/*
	@echo "✅ Données brutes supprimées"

clean: clean-cache clean-checkpoints clean-results ## Nettoie cache + checkpoints + résultats
	@echo "🧹 Nettoyage terminé"

clean-all: clean clean-data ## Nettoie TOUT (y compris données brutes)
	@echo "🧹 Nettoyage complet terminé"

# === Dépendances ===
install: ## Installe les dépendances Python
	pip install -r requirements.txt
