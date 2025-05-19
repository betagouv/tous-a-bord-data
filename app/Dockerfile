FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système et Chromium
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    chromium \
    chromium-driver \
    libgconf-2-4 \
    libxss1 \
    libnss3 \
    libnspr4 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    xvfb \
    unzip \
    # Ajout des dépendances pour crawl4ai
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Installation de Playwright et ses navigateurs
RUN npm install -g playwright
RUN npx playwright install
RUN npx playwright install-deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installation de spaCy et du modèle français
RUN pip install --no-cache-dir spacy
RUN python -m spacy download fr_core_news_lg

# Installation de LlamaIndex et ses dépendances
RUN pip install --no-cache-dir llama-index llama-index-core llama-index-llms-anthropic llama-index-llms-ollama

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# --server.fileWatcherType=none fix spacy torch error
ENTRYPOINT ["streamlit", "run", "🚎_Critères_d'éligibilité_par_AOM.py", "--server.port=8501", "--server.address=0.0.0.0"] 