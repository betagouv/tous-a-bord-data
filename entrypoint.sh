#!/bin/bash

# Si la variable BATCH_MODE est définie, exécuter le traitement batch
if [ -n "$BATCH_MODE" ]; then
    python -c "import os; from scripts.run_tag_pipeline import main; main(os.getenv('SIREN'))"
else
    # Sinon, lancer le serveur Streamlit normal
    streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0
fi
