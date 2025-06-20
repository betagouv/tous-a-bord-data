import os

# import streamlit as st
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
}


def get_model_metadata(model_name: str, platform: str) -> dict:
    """Crée les métadonnées pour le tracking LangSmith"""
    return f"model: {model_name}, platform: {platform}"


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
