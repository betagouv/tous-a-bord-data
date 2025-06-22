FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN crawl4ai-setup && apt-get remove -y gcc g++ build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && python -m spacy download fr_core_news_md

COPY . .

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Définir le nouvel entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]