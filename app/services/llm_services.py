import os

import ollama
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv


# OLLAMA with small models
def ensure_ollama_host():
    if "OLLAMA_HOST" not in os.environ:
        load_dotenv()
        ollama_host = os.getenv("OLLAMA_HOST")
        if ollama_host:
            os.environ["OLLAMA_HOST"] = ollama_host


def call_ollama(prompt, model="llama3:8b"):
    ensure_ollama_host()
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ANTHROPIC for an access to the best models
class AnthropicWrapper:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnthropicWrapper, cls).__new__(cls)
            # Création du client une seule fois
            cls._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return cls._instance

    def stream_anthropic(self, **kwargs):
        """
        Lance un stream avec le client singleton.
        kwargs : tous les arguments attendus par client.messages.create
        """
        return self._client.messages.create(**kwargs)


def call_anthropic(prompt, model):
    client = AnthropicWrapper()
    stream = client.stream_anthropic(
        model=model,
        max_tokens=1024,  # max tokens for output
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    st.write(f"---> 3 {model}")
    current_chunk_text = ""
    for event in stream:
        if event.type == "content_block_delta":
            current_chunk_text += event.delta.text
    return current_chunk_text


# TODO: add openai client for scaleways models
