import streamlit as st
from services.nlp_services import (
    count_tokens,
    extract_markdown_text,
    filter_text_with_spacy,
    load_spacy_model,
    normalize_text,
)

# Interface Streamlit
st.title("Détection avancée de critères d'éligibilité avec SpaCy")

# Chargement du modèle SpaCy
with st.spinner("Chargement et configuration du modèle linguistique..."):
    nlp = load_spacy_model()
    st.success("Modèle SpaCy configuré avec succès !")

# Zone de texte pour tester
text = st.text_area("Entrez un texte Markdown à analyser :", height=250)

if text:
    with st.spinner("Analyse en cours..."):
        raw_text = extract_markdown_text(text)
        paragraphs = normalize_text(raw_text, nlp)
        paragraphs_filtered, relevant_sentences = filter_text_with_spacy(
            paragraphs, nlp
        )
        # Concaténation des paragraphes filtrés en une seule chaîne
        paragraphes_concatenes = "\n\n".join(paragraphs_filtered)
        # Comptage du nombre de tokens
        nb_tokens = count_tokens(paragraphes_concatenes, nlp)
        title_part1 = "Paragraphes contenant des critères d'éligibilité"
        title = f"{title_part1} ({nb_tokens} tokens)"
        st.subheader(title)
        st.text_area(
            "Tous les paragraphes filtrés", paragraphes_concatenes, height=300
        )
