import os

import ollama

# import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

MAX_TOKEN_OUTPUT = 4000

LLM_MODELS = {
    "Llama 3.1 8B (Scaleway)": {
        "name": "llama-3.1-8b-instruct",
        "max_tokens": 128000,
    },
    "Llama 3.3 70B (Scaleway)": {
        "name": "llama-3.3-70b-instruct",
        "max_tokens": 131000,
    },
    "Mistral Nemo (Scaleway)": {
        "name": "mistral-nemo-instruct-2407",
        "max_tokens": 128000,
    },
    "Qwen 2.5 32B (Scaleway)": {
        "name": "qwen2.5-coder-32b-instruct",
        "max_tokens": 32000,
    },
    # not really supported yet
    # "DeepSeek r1 (Scaleway)": {
    #     "name": "deepseek-r1",
    #     "max_tokens": 20000,
    # },
    "DeepSeek r1 distill (Scaleway)": {
        "name": "deepseek-r1-distill-llama-70b",
        "max_tokens": 32000,
    },
    "Claude 3 Haiku (Anthropic)": {
        "name": "claude-3-5-haiku-latest",
        "max_tokens": 100000,
    },
    # too expansive
    "Claude 3 Sonnet (Anthropic)": {
        "name": "claude-3-5-sonnet-latest",
        "max_tokens": 200000,
    },
    "Claude 4 Sonnet (Anthropic)": {
        "name": "claude-sonnet-4-20250514",
        "max_tokens": 200000,
    },
}


def get_model_metadata(model_name: str, platform: str) -> dict:
    """Crée les métadonnées pour le tracking LangSmith"""
    return f"model: {model_name}, platform: {platform}"


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
        max_tokens=MAX_TOKEN_OUTPUT,  # max tokens for output
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    current_chunk_text = ""
    for event in stream:
        if event.type == "content_block_delta":
            current_chunk_text += event.delta.text
    return current_chunk_text


# TODO: add openai client for scaleways models
def call_scaleway(prompt, model):
    client = OpenAI(
        base_url=os.getenv(
            "SCALEWAY_API_URL"
        ),  # Scaleway's Generative APIs service URL
        api_key=os.getenv(
            "SCALEWAY_API_KEY"
        ),  # Your unique API key from Scaleway
    )
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKEN_OUTPUT,  # max tokens for output
        stream=True,
    )
    current_chunk_text = ""
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            current_chunk_text += event.choices[0].delta.content
    return current_chunk_text
