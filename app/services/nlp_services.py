import re
from typing import Dict, List

import markdown
import spacy
import spacy.util
import streamlit as st
from bs4 import BeautifulSoup
from constants.entites_eligibilite import ENTITES
from constants.tag_dp_mapping import TAG_DP_MAPPING
from constants.tokens_eligibilite import TOKENS_ELIGIBILITE
from spacy.matcher import Matcher, PhraseMatcher

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
    """Crée un matcher pour les critères de transport"""
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

    return phrase_matcher, matcher


def extract_transport_fare(text: str, nlp) -> List[Dict[str, str]]:
    """Extrait les critères de transport du texte"""
    # Créer les matchers
    phrase_matcher, matcher = create_transport_fare_matcher(nlp)

    # Traiter le texte
    doc = nlp(text)

    # Initialiser la liste des résultats
    results = []

    # Utiliser le phrase matcher
    matches = phrase_matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        results.append(
            {
                "type": "CRITERE_ELIGIBILITE",
                "text": span.text,
                "lemma": span.lemma_,
            }
        )

    # Utiliser le regex matcher
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]
        span = doc[start:end]
        results.append({"type": rule_id, "text": span.text})

    # Reconnaître les entités spécifiques
    for token in doc:
        if token.text in ENTITES:
            results.append(
                {"type": "ACRONYME_ELIGIBILITE", "text": token.text}
            )

    return results


# Load the fr_core_news_lg model
@st.cache_resource
def load_spacy_model():
    """Charge le modèle SpaCy de base"""
    try:
        nlp = spacy.load("fr_core_news_lg")
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


def filter_transport_fare(paragraphs, nlp):
    """Filter paragraphs to keep only those with eligibility criteria"""
    filtered_paragraphs = []
    relevant_sentences = []

    # Créer les matchers une seule fois
    phrase_matcher, matcher = create_transport_fare_matcher(nlp)

    for paragraph in paragraphs:
        doc = nlp(paragraph)

        # Chercher les critères dans tout le paragraphe
        matches_phrase = phrase_matcher(doc)
        matches_regex = matcher(doc)
        matches_entites = any(token.text in ENTITES for token in doc)

        # Si on trouve au moins un match, le paragraphe est pertinent
        if matches_phrase or matches_regex or matches_entites:
            filtered_paragraphs.append(paragraph)
            # Ajouter les phrases qui contiennent les matches
            for sent in doc.sents:
                sent_start = sent.start
                sent_end = sent.end

                # Vérifier si la phrase contient un match du phrase_matcher
                has_phrase_match = any(
                    sent_start <= start < sent_end
                    for _, start, _ in matches_phrase
                )

                # Vérifier si la phrase contient un match du regex_matcher
                has_regex_match = any(
                    sent_start <= start < sent_end
                    for _, start, _ in matches_regex
                )

                # Vérifier si la phrase contient une entité
                has_entity = any(token.text in ENTITES for token in sent)

                if has_phrase_match or has_regex_match or has_entity:
                    relevant_sentences.append(sent.text)

    return filtered_paragraphs, relevant_sentences


def count_tokens(text, nlp):
    """Compte le nombre de tokens dans un texte en utilisant SpaCy"""
    doc = nlp(text)
    return len(doc)


def create_tag_matcher(nlp):
    """Crée un matcher uniquement pour les tags"""
    # Configure the phrase matcher for tags
    tag_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Add patterns for each token in TAG_DP_MAPPING
    for token in TAG_DP_MAPPING.keys():
        pattern = nlp(token.lower())
        tag_info = TAG_DP_MAPPING[token]
        # Use the tag as the pattern name
        tag_matcher.add(tag_info["tag"], [pattern])

    return tag_matcher


def create_dp_matcher(nlp):
    """Crée un matcher uniquement pour les data providers"""
    # Configure the phrase matcher for data providers
    dp_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Add patterns for each token in TAG_DP_MAPPING
    for token in TAG_DP_MAPPING.keys():
        pattern = nlp(token.lower())
        dp_info = TAG_DP_MAPPING[token]
        # Use the provider as the pattern name
        dp_matcher.add(dp_info["fournisseur"], [pattern])

    return dp_matcher


def extract_tags(text: str, nlp) -> List[str]:
    """Extrait uniquement les tags du texte"""
    # Créer le matcher
    tag_matcher = create_tag_matcher(nlp)

    # Traiter le texte
    doc = nlp(text)

    # Trouver les correspondances
    matches = tag_matcher(doc)

    # Collecter les tags uniques
    tags = []
    for match_id, start, end in matches:
        tag_name = nlp.vocab.strings[match_id]
        if tag_name not in tags:
            tags.append(tag_name)

    return sorted(tags)


def extract_data_providers(text: str, nlp) -> List[str]:
    """Extrait uniquement les fournisseurs de données du texte"""
    # Créer le matcher
    dp_matcher = create_dp_matcher(nlp)

    # Traiter le texte
    doc = nlp(text)

    # Trouver les correspondances
    matches = dp_matcher(doc)

    # Collecter les fournisseurs uniques
    providers = []
    for match_id, start, end in matches:
        provider_name = nlp.vocab.strings[match_id]
        if provider_name not in providers:
            providers.append(provider_name)

    return sorted(providers)
