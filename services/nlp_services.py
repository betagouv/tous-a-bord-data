import re

import markdown
import spacy
import streamlit as st
from bs4 import BeautifulSoup
from spacy.matcher import Matcher, PhraseMatcher

from constants.entites_eligibilite import ENTITES
from constants.tag_dp_mapping import BLACK_LIST, TAG_DP_MAPPING
from constants.tokens_eligibilite import TOKENS_ELIGIBILITE

# commons
TOKENS_ELIGIBILITE = [token.lower() for token in TOKENS_ELIGIBILITE]


# Load the fr_core_news_sm model (lightest French model)
@st.cache_resource
def load_spacy_model():
    """Charge le modèle SpaCy de base (version légère)"""
    try:
        nlp = spacy.load("fr_core_news_sm")
        return nlp
    except OSError:
        st.error("Installation en cours...")
        import subprocess

        subprocess.run(
            ["python", "-m", "spacy", "download", "fr_core_news_sm"]
        )
        return spacy.load("fr_core_news_sm")


@st.cache_data
def get_cached_mapping_lemmas():
    """Cache le mapping lemmatisé pour éviter de le recalculer"""
    nlp = load_spacy_model()
    tag_dp_mapping_lemmas = {}

    for k, v in TAG_DP_MAPPING.items():
        if k and v:
            doc_key = nlp(k.lower().replace("'", "'"))
            lemmas = [token.lemma_.lower() for token in doc_key]
            lemma_key = " ".join(lemmas)
            tag_dp_mapping_lemmas[lemma_key] = v

    return tag_dp_mapping_lemmas


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


def count_tokens(text, nlp):
    """Compte le nombre de tokens dans un texte en utilisant SpaCy"""
    doc = nlp(text)
    return len(doc)


def create_eligibility_matcher(nlp):
    """Crée un matcher pour détecter les tokens en utilisant la lemmisation"""
    # Configure the phrase matcher for the eligibility criteria
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    # Add the patterns for the eligibility terms
    patterns = []

    for text in TOKENS_ELIGIBILITE:
        doc = nlp(text.lower())
        patterns.append(doc)

    phrase_matcher.add("CRITERE_ELIGIBILITE", patterns)

    # Configure the regex matcher for the patterns
    matcher = Matcher(nlp.vocab)

    # Add entites to the matcher
    # Note: multi-tokens ENTITES are handled
    for entite in ENTITES:
        tokens = entite.split()
        if len(tokens) > 1:
            pattern = []
            for token in tokens:
                pattern.append({"LOWER": token.lower()})
        else:
            # monotoken ENTITES
            pattern = [{"TEXT": entite}]

        matcher.add(f"ENTITE_{entite}", [pattern])

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


def create_transport_fare_matcher(nlp):
    """Crée un matcher pour les critères de transport"""
    # Retrieve the eligibility matcher
    matcher = Matcher(nlp.vocab)

    # Add fares patterns to the matcher
    matcher.add(
        "TARIF",
        [
            # Detect : "10€"
            [{"TEXT": {"REGEX": r"\d+"}}, {"TEXT": "€"}],
            # Detect : "10 euros"
            [{"TEXT": {"REGEX": r"\d+"}}, {"TEXT": "euros"}],
            # Detect : "10 %"
            # [{"TEXT": {"REGEX": r"\d+"}}, {"TEXT": "%"}],
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

    return matcher


def filter_transport_fare(paragraphs, nlp):
    """Filter paragraphs to keep only those with eligibility criteria"""
    filtered_paragraphs = []
    relevant_sentences = []

    (
        eligibility_phrase_matcher,
        eligibility_matcher,
    ) = create_eligibility_matcher(nlp)
    fare_matcher = create_transport_fare_matcher(nlp)

    for paragraph in paragraphs:
        doc = nlp(paragraph)

        eligibility_matches_phrase = eligibility_phrase_matcher(doc)
        eligibility_matches_regex = eligibility_matcher(doc)
        fare_matches_regex = fare_matcher(doc)

        matches_entites = False
        for match_id, start, _ in eligibility_matches_regex:
            match_type = nlp.vocab.strings[match_id]
            if match_type.startswith("ENTITE_"):
                matches_entites = True
                break

        # Paragraphs with matches
        if (
            eligibility_matches_phrase
            or eligibility_matches_regex
            or matches_entites
            or fare_matches_regex
        ):
            filtered_paragraphs.append(paragraph)
            # Add sentences with matches
            for sent in doc.sents:
                sent_start = sent.start
                sent_end = sent.end

                has_eligibility_phrase_match = any(
                    sent_start <= start < sent_end
                    for _, start, _ in eligibility_matches_regex
                )

                has_eligibility_regex_match = any(
                    sent_start <= start < sent_end
                    for _, start, _ in eligibility_matches_regex
                )

                has_eligibility_entity = False
                for match_id, start, end in eligibility_matches_regex:
                    match_type = nlp.vocab.strings[match_id]
                    if (
                        match_type.startswith("ENTITE_")
                        and sent_start <= start < sent_end
                    ):
                        has_eligibility_entity = True
                        break

                has_fare_regex_match = any(
                    sent_start <= start < sent_end
                    for _, start, _ in fare_matches_regex
                )

                if (
                    has_eligibility_phrase_match
                    or has_eligibility_regex_match
                    or has_eligibility_entity
                    or has_fare_regex_match
                ):
                    relevant_sentences.append(sent.text)

    return filtered_paragraphs, relevant_sentences


def get_matches_and_lemmas(text: str, nlp) -> tuple:
    """Extrait les matches et les lemmes à partir du texte."""

    phrase_matcher, matcher = create_eligibility_matcher(nlp)
    text = text.replace("'", "'")
    doc = nlp(text)

    matches_phrase = phrase_matcher(doc)
    matches = matcher(doc)

    matches_entites = False
    for match_id, _, _ in matches:
        match_type = nlp.vocab.strings[match_id]
        if match_type.startswith("ENTITE_"):
            matches_entites = True
            break

    tag_dp_mapping_lemmas = get_cached_mapping_lemmas()

    return doc, matches_phrase, matches_entites, matches, tag_dp_mapping_lemmas


def get_highlighted_sentence(doc, start, end, start_char=None, text=None):
    """Trouve et met en surbrillance une partie de phrase."""
    sent = next(
        (
            sent
            for sent in doc.sents
            if start >= sent.start and end <= sent.end
        ),
        None,
    )

    if not sent:
        return text if text else doc[start:end].text

    if start_char is None:
        start_char = doc[start].idx
        end_char = doc[end - 1].idx + len(doc[end - 1].text)
    else:
        end_char = start_char + len(text)

    before = sent.text[: start_char - sent.start_char]
    match = (
        text
        if text
        else sent.text[
            start_char - sent.start_char : end_char - sent.start_char
        ]
    )
    after = sent.text[end_char - sent.start_char :]

    return f"{before}<mark>{match}</mark>{after}"


def extract_from_matches(
    doc,
    matches_phrase,
    matches_entites,
    matches,
    tag_dp_mapping_lemmas,
    nlp,
    field,
) -> tuple:
    """Extrait les valeurs uniques et les matches de debug."""
    valeurs_uniques = set()
    debug_matches = {}

    # Helper function to check if a span is blacklisted
    def is_blacklisted(start, end):
        context_window = 5
        start_context = max(0, start - context_window)
        end_context = min(len(doc), end + context_window)
        context_span = doc[start_context:end_context].text.lower()
        context_span = re.sub(r"\s+", " ", context_span).strip()

        # Check if black listed word is present as a whole word
        # not as a subchain
        for black_term in BLACK_LIST:
            pattern = r"\b" + re.escape(black_term) + r"\b"
            if re.search(pattern, context_span, re.IGNORECASE):
                print(
                    "⚠️ Span ignoré car contient le mot entier",
                    f"'{black_term}' dans: {context_span}",
                )
                return True
        return False

    for match_id, start, end in matches_phrase:
        span = doc[start:end]
        if not span.text:
            continue

        if is_blacklisted(start, end):
            continue

        span_lemmas = [token.lemma_.lower() for token in span]
        span_text = " ".join(span_lemmas)

        if span_text in tag_dp_mapping_lemmas:
            valeur = tag_dp_mapping_lemmas[span_text].get(field)
            if valeur and valeur not in debug_matches:
                valeurs_uniques.add(valeur)
                debug_matches[valeur] = get_highlighted_sentence(
                    doc, span.start, span.end
                )

    if matches_entites:
        for match_id, start, end in matches:
            match_type = nlp.vocab.strings[match_id]

            if match_type.startswith("ENTITE_"):
                # Check blacklist for entity matches
                if is_blacklisted(start, end):
                    continue

                entite = match_type[7:]

                entity_mapping = None
                for k, v in TAG_DP_MAPPING.items():
                    if k and k.lower() == entite.lower():
                        entity_mapping = v
                        break

                if entity_mapping:
                    valeur = entity_mapping.get(field)
                    if valeur and valeur not in debug_matches:
                        print(
                            "✅ Entité détectée comme mot entier: "
                            f"'{entite}' -> '{valeur}'"
                        )
                        valeurs_uniques.add(valeur)
                        debug_matches[valeur] = get_highlighted_sentence(
                            doc, start, end
                        )

    # For special matches (AGE and QF)
    if field == "tag":
        special_tags = {"AGE": "Age", "QF": "Quotient Familial"}
        for match_id, start, end in matches:
            match_type = nlp.vocab.strings[match_id]
            if match_type in special_tags:
                # Check blacklist for special pattern matches
                if is_blacklisted(start, end):
                    continue

                tag = special_tags[match_type]
                if tag and tag not in debug_matches:
                    valeurs_uniques.add(tag)
                    debug_matches[tag] = get_highlighted_sentence(
                        doc, start, end
                    )

    return valeurs_uniques, debug_matches


def extract_tags_and_providers(
    text: str, nlp, siren: str = None, name: str = None
) -> tuple:
    """Extrait les tags ET les fournisseurs en une seule passe optimisée"""

    (
        doc,
        matches_phrase,
        matches_entites,
        matches,
        tag_dp_mapping_lemmas,
    ) = get_matches_and_lemmas(text, nlp)

    tags_uniques, tags_debug = extract_from_matches(
        doc,
        matches_phrase,
        matches_entites,
        matches,
        tag_dp_mapping_lemmas,
        nlp,
        "tag",
    )

    providers_uniques, providers_debug = extract_from_matches(
        doc,
        matches_phrase,
        matches_entites,
        matches,
        tag_dp_mapping_lemmas,
        nlp,
        "fournisseur",
    )

    return (
        sorted(list(tag for tag in tags_uniques if tag is not None)),
        sorted(
            list(
                provider
                for provider in providers_uniques
                if provider is not None
            )
        ),
        tags_debug,
        providers_debug,
    )
