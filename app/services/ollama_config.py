import os

from dotenv import load_dotenv


def ensure_ollama_host():
    if "OLLAMA_HOST" not in os.environ:
        load_dotenv()
        ollama_host = os.getenv("OLLAMA_HOST")
        if ollama_host:
            os.environ["OLLAMA_HOST"] = ollama_host
