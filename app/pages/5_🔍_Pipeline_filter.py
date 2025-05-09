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


def get_sentence_embeddings(doc, nlp):
    """Extraire les embeddings pour chaque phrase"""
    # List for storing phrases and their embeddings
    sentence_embeddings = []
    for sent in doc.sents:
        # Create a separate Doc object for the phrase
        sent_doc = nlp(sent.text)
        # Get the embedding of the phrase (average of token embeddings)
        embedding = sent_doc.vector
        # Add to the list
        sentence_embeddings.append({"text": sent.text, "embedding": embedding})
    return sentence_embeddings


# Load the fr_core_news_lg model
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
with st.spinner("Chargement du modèle linguistique fr_core_news_lg..."):
    nlp = load_spacy_model()
    st.success("Modèle chargé avec succès !")

# Text area to test
text = st.text_area("Enter text to analyze:", height=150)

if text:
    with st.spinner("Analyse en cours..."):
        text_raw = extract_text_markdown(text)
        doc = nlp(text_raw)
        # Segment into sentences
        for i, sent in enumerate(doc.sents):
            st.write(f"{sent.text}")
            sentence_embeddings = get_sentence_embeddings(doc, nlp)
            st.write(sentence_embeddings)
