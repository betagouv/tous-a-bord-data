import re

import markdown
import spacy
import spacy.util
import streamlit as st
from bs4 import BeautifulSoup
from constants.entites_eligibilite import ENTITES
from constants.tokens_eligibilite import TOKENS_ELIGIBILITE
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span

TOKENS_ELIGIBILITE = [token.lower() for token in TOKENS_ELIGIBILITE]


def extract_markdown_text(markdown_text):
    """Convertit le Markdown en texte brut"""
    if not markdown_text:
        return ""
    # Convert Markdown to HTML
    html = markdown.markdown(markdown_text)
    # Extract the text from the HTML
    soup = BeautifulSoup(html, "html.parser")
    raw_text = soup.get_text("\n")
    return raw_text


def create_transport_fare_matcher(nlp):
    """Crée et configure le composant transport_fare_matcher"""

    @spacy.Language.component("transport_fare_matcher")
    def matcher(doc):
        # Use the phrase matcher
        matches = nlp.user_data["phrase_matcher"](doc)
        for match_id, start, end in matches:
            # Create a span with a custom label
            span = Span(doc, start, end, label="CRITERE_ELIGIBILITE")
            doc.ents = spacy.util.filter_spans(list(doc.ents) + [span])
        # Use the regex matcher
        matches = nlp.user_data["matcher"](doc)
        for match_id, start, end in matches:
            # Get the name of the pattern
            rule_id = nlp.vocab.strings[match_id]
            span = Span(doc, start, end, label=rule_id)
            doc.ents = spacy.util.filter_spans(list(doc.ents) + [span])
        # Recognize specific entities
        for token in doc:
            if token.text in ENTITES:
                span = Span(
                    doc, token.i, token.i + 1, label="ACRONYME_ELIGIBILITE"
                )
                doc.ents = spacy.util.filter_spans(list(doc.ents) + [span])
        return doc

    return matcher


# Load the fr_core_news_lg model
@st.cache_resource
def load_spacy_model():
    """Charge le modèle SpaCy et configure les composants personnalisés"""
    try:
        nlp = spacy.load("fr_core_news_lg")
        # Initialiser user_data s'il n'existe pas déjà
        if not hasattr(nlp, "user_data"):
            nlp.user_data = {}
        # Configure the phrase matcher for the eligibility criteria
        phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
        # Add the patterns for the eligibility terms
        patterns = [nlp(text) for text in TOKENS_ELIGIBILITE]
        phrase_matcher.add("CRITERE_ELIGIBILITE", patterns)
        # Configure the regex matcher for the patterns
        matcher = Matcher(nlp.vocab)
        # Patterns for ages
        matcher.add(
            "AGE",
            [
                # Detect : "18 ans", "25 ans", etc.
                [{"TEXT": {"REGEX": r"\d+"}}, {"LOWER": "ans"}],
                # Detect : "18 ans et plus", "25 ans et plus", etc.
                [
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "ans"},
                    {"LOWER": "et"},
                    {"LOWER": "plus"},
                ],
                # Detect : "moins de 18 ans"
                [
                    {"LOWER": "moins"},
                    {"LOWER": "de"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "ans"},
                ],
                # Detect : "plus de 18 ans"
                [
                    {"LOWER": "plus"},
                    {"LOWER": "de"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "ans"},
                ],
                # Detect : "entre 18 et 25 ans"
                [
                    {"LOWER": "entre"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "et"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "ans"},
                ],
                # Detect : "de 18 à 25 ans"
                [
                    {"LOWER": "de"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "à"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "ans"},
                ],
                # Detect : "à partir de 18 ans"
                [
                    {"LOWER": "à"},
                    {"LOWER": "partir"},
                    {"LOWER": "de"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "ans"},
                ],
                # Detect : "18/25 ans"
                [
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"TEXT": "/"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "ans"},
                ],
            ],
        )
        # Patterns pour montants et pourcentages
        matcher.add(
            "TARIF",
            [
                # Detect : "10€"
                [{"TEXT": {"REGEX": r"\d+"}}, {"TEXT": "€"}],
                # Detect : "10 euros"
                [{"TEXT": {"REGEX": r"\d+"}}, {"TEXT": "euros"}],
                # Detect : "10 %"
                [{"TEXT": {"REGEX": r"\d+"}}, {"TEXT": "%"}],
                # Detect : "10 €/an"
                [
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"TEXT": "€"},
                    {"LOWER": "/"},
                    {"LOWER": {"IN": ["an", "mois", "jour"]}},
                ],
                # Detect : "10 euros/an"
                [
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"TEXT": "euros"},
                    {"LOWER": "par"},
                    {"LOWER": {"IN": ["an", "mois", "jour"]}},
                ],
            ],
        )
        # Patterns pour quotient familial
        matcher.add(
            "QF",
            [
                # "quotient familial inférieur à 1",
                # "quotient familial supérieur à 1"
                [
                    {"LOWER": {"IN": ["qf", "quotient familial"]}},
                    {
                        "LOWER": {
                            "IN": [
                                "inférieur",
                                "supérieur",
                                ">",
                                "<",
                                ">=",
                                "<=",
                            ]
                        }
                    },
                    {"LOWER": "à"},
                    {"TEXT": {"REGEX": r"\d+"}},
                ],
                # Detect : "QF de 1 à 2"
                [
                    {"LOWER": {"IN": ["qf", "quotient familial"]}},
                    {"LOWER": "de"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "à"},
                    {"TEXT": {"REGEX": r"\d+"}},
                ],
                # Detect : "QF entre 1 et 2"
                [
                    {"LOWER": {"IN": ["qf", "quotient familial"]}},
                    {"LOWER": "entre"},
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": "et"},
                    {"TEXT": {"REGEX": r"\d+"}},
                ],
            ],
        )
        # Add the matcher and the phrase_matcher to the pipeline components
        nlp.user_data["matcher"] = matcher
        nlp.user_data["phrase_matcher"] = phrase_matcher

        # Create and add the eligibilite_matcher component
        # flake8: noqa: F841
        transport_fare_matcher = create_transport_fare_matcher(nlp)
        nlp.add_pipe("transport_fare_matcher", last=True)
        return nlp
    except OSError:
        st.error("Installation en cours...")
        import subprocess

        subprocess.run(
            ["python", "-m", "spacy", "download", "fr_core_news_lg"]
        )
        return spacy.load("fr_core_news_lg")


def normalize_text(raw_text, nlp):
    """Use SpaCy to normalize text (clean and structure it)"""
    # Process text with SpaCy
    doc = nlp(raw_text)
    # Use SpaCy's sentence segmentation
    paragraphs = []
    current_paragraph = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        # Skip empty sentences or those with only special characters
        if not sent_text or re.match(r"^[-*#=_\s]+$", sent_text):
            continue
        if len(sent_text) > 0:
            current_paragraph.append(sent_text)
        # Detect paragraph end (empty line, newline)
        if sent.text.endswith("\n\n") or sent.text.endswith("\r\n\r\n"):
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
    # Add the last paragraph if not empty
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))
    # If no paragraphs detected, create one per sentence
    if not paragraphs and doc.sents:
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text and not re.match(r"^[-*#=_\s]+$", sent_text):
                paragraphs.append(sent_text)
    return paragraphs


def phrase_contains_criterion(sent, nlp):
    """Use SpaCy's capabilities to detect eligibility criteria in a sentence"""
    # Analyze the sentence with SpaCy
    doc = nlp(sent.text if hasattr(sent, "text") else sent)
    # Check the recognized entities
    for ent in doc.ents:
        if ent.label_ in [
            "CRITERE_ELIGIBILITE",
            "AGE",
            "TARIF",
            "QF",
            "ACRONYME_ELIGIBILITE",
        ]:
            return True
    return False


def filter_text_with_spacy(paragraphs, nlp):
    """Filter paragraphs to keep only those with eligibility criteria"""
    filtered_paragraphs = []
    relevant_sentences = []
    for paragraph in paragraphs:
        doc = nlp(paragraph)
        # Check each sentence in the paragraph
        contains_criterion = False
        for sent in doc.sents:
            if phrase_contains_criterion(sent, nlp):
                contains_criterion = True
                relevant_sentences.append(sent.text)
        if contains_criterion:
            filtered_paragraphs.append(paragraph)
    return filtered_paragraphs, relevant_sentences


def count_tokens(text, nlp):
    """Compte le nombre de tokens dans un texte en utilisant SpaCy"""
    doc = nlp(text)
    return len(doc)
