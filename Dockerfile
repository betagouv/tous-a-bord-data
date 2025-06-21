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

EXPOSE ${PORT:-8501}

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health

ENTRYPOINT ["sh", "-c", "streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
