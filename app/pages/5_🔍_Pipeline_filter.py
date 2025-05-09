import markdown
import numpy as np
import pandas as pd
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


# Initialiser la session state pour stocker les critères d'éligibilité
if "eligibility_criteria" not in st.session_state:
    st.session_state.eligibility_criteria = []


if "criteria_embeddings" not in st.session_state:
    st.session_state.criteria_embeddings = []

# Initialiser le seuil de similarité
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.6


st.title("Pipeline de filtrage avec SpaCy")
with st.spinner("Chargement du modèle linguistique fr_core_news_lg..."):
    nlp = load_spacy_model()
    st.success("Modèle chargé avec succès !")

# Ajouter un text area pour les critères d'éligibilité
st.subheader("Critères d'éligibilité")
criteria_text = st.text_area(
    "Entrez les critères d'éligibilité (un par ligne):",
    height=150,
    help="Ces critères seront utilisés pour filtrer les phrases pertinentes",
)

# Bouton pour ajouter les critères à la session
if st.button("Sauvegarder les critères"):
    if criteria_text:
        # Diviser le texte en lignes et filtrer les lignes vides
        criteria_list = [
            line.strip() for line in criteria_text.split("\n") if line.strip()
        ]

        # Sauvegarder dans la session state
        st.session_state.eligibility_criteria = criteria_list

        # Générer les embeddings pour chaque critère
        criteria_embeddings = []
        for criterion in criteria_list:
            doc = nlp(criterion)
            criteria_embeddings.append(
                {"text": criterion, "embedding": doc.vector}
            )

        st.session_state.criteria_embeddings = criteria_embeddings

        st.success(f"✅ {len(criteria_list)} critères sauvegardés!")
    else:
        st.warning("Aucun critère à sauvegarder.")

# Afficher les critères sauvegardés
if st.session_state.eligibility_criteria:
    with st.expander("Critères sauvegardés", expanded=True):
        for i, criterion in enumerate(st.session_state.eligibility_criteria):
            st.write(f"{i+1}. {criterion}")

# Ajouter un slider pour le seuil de similarité
st.subheader("Configuration du filtre")
similarity_threshold = st.slider(
    "Seuil de similarité :",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.similarity_threshold,
    step=0.05,
    help="Ne conserver que les phrases similaires à plus de X%",
)
# Mettre à jour la session state avec la nouvelle valeur
st.session_state.similarity_threshold = similarity_threshold

# Text area to test
st.subheader("Analyse de texte")
text = st.text_area("Entrez le texte à analyser:", height=150)

if text:
    with st.spinner("Analyse en cours..."):
        text_raw = extract_text_markdown(text)
        doc = nlp(text_raw)

        # Segment into sentences and calculate embeddings
        sentence_embeddings = get_sentence_embeddings(doc, nlp)

        # Créer une liste pour stocker les phrases pertinentes
        relevant_sentences = []

        # Pour compter les phrases filtrées
        filtered_count = 0
        total_count = len(sentence_embeddings)

        for i, sent_data in enumerate(sentence_embeddings):
            # Si des critères ont été sauvegardés, calculer la similarité
            if st.session_state.criteria_embeddings:
                max_similarity = 0
                best_criterion = ""

                for criterion_data in st.session_state.criteria_embeddings:
                    similarity = np.dot(
                        sent_data["embedding"], criterion_data["embedding"]
                    ) / (
                        np.linalg.norm(sent_data["embedding"])
                        * np.linalg.norm(criterion_data["embedding"])
                    )

                    # Mettre à jour la meilleure similarité
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_criterion = criterion_data["text"]

                if max_similarity >= similarity_threshold:
                    filtered_count += 1
                    relevant_sentences.append(
                        {
                            "num": i + 1,
                            "phrase": sent_data["text"],
                            "critère": best_criterion,
                            "similarité": max_similarity,
                        }
                    )

        # Afficher un résumé des phrases filtrées
        st.subheader("Résumé du filtrage")
        if total_count > 0:
            st.write(
                f"**{filtered_count}** phrases pertinentes sur {total_count}"
                f"({filtered_count/total_count:.0%})"
            )
        else:
            st.write("Aucune phrase détectée dans le texte.")

        # Afficher uniquement les phrases pertinentes
        if relevant_sentences:
            st.subheader("✅ Phrases pertinentes")

            # Créer un dataframe pour un affichage alternatif
            with st.expander("Affichage tabulaire"):
                df_relevant = pd.DataFrame(relevant_sentences)
                st.dataframe(
                    df_relevant.style.format({"similarité": "{:.2f}"})
                )

                # Ajouter un bouton pour télécharger les résultats
                csv = df_relevant.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger les résultats (CSV)",
                    csv,
                    "phrases_pertinentes.csv",
                    "text/csv",
                    key="download-csv",
                )
        else:
            st.info("Aucune phrase ne correspond aux critères d'éligibilité")
