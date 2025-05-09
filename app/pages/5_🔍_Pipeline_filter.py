import markdown
import spacy
import streamlit as st
from bs4 import BeautifulSoup


def extract_text_markdown(text_markdown):
    """Convert markdown to text"""
    html = markdown.markdown(text_markdown)
    # Extract text from HTML
    soup = BeautifulSoup(html, "html.parser")
    text_raw = soup.get_text("\n")
    return text_raw


# Load the fr_dep_news_trf model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("fr_core_news_lg")
    except OSError:
        st.error("Installation en cours...")
        import subprocess

        subprocess.run(
            ["python", "-m", "spacy", "download", "fr_core_news_lg"]
        )
        return spacy.load("fr_core_news_lg")


st.title("Pipeline de filtrage avec SpaCy")

# Chargement du modèle
with st.spinner("Chargement du modèle linguistique fr_dep_news_trf..."):
    nlp = load_spacy_model()
    st.success("Modèle chargé avec succès !")

# Zone de texte pour tester
text = st.text_area("Entrez un texte à analyser :", height=150)

if text:
    with st.spinner("Analyse en cours..."):
        text_raw = extract_text_markdown(text)
        doc = nlp(text_raw)
        # Segmenter en phrases
        for i, sent in enumerate(doc.sents):
            st.write(f"{sent.text}")
