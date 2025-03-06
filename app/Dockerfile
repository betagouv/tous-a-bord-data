FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système pour Playwright/Chromium
RUN apt-get update && apt-get install -y \
    wget gnupg libgconf-2-4 libxss1 libnss3 libnspr4 libasound2 \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdbus-1-3 libdrm2 \
    libgbm1 libgtk-3-0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installer les dépendances Python et Playwright en une seule couche
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir playwright && \
    playwright install chromium && \
    playwright install-deps

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Accueil.py", "--server.port=8501", "--server.address=0.0.0.0"] 