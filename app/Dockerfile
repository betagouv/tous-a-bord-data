FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    chromium \
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
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installation de Crawl4AI et configuration automatique
RUN pip install --no-cache-dir "crawl4ai>=0.5.0,<0.6.0rc1"
RUN crawl4ai-setup


# Installation de spaCy et du modèle français
RUN python -m spacy download fr_core_news_lg

COPY . .

# Expose the port that will be used by Streamlit
# For Scalingo, we'll use the PORT environment variable
EXPOSE ${PORT:-8501}

# Health check using the appropriate port
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health
# --server.fileWatcherType=none fix spacy torch error
# Use PORT environment variable for Scalingo compatibility
ENTRYPOINT ["sh", "-c", "streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
