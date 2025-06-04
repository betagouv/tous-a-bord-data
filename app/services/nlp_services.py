import re

import markdown
import spacy
import spacy.util
import streamlit as st
from bs4 import BeautifulSoup
from constants.entites_eligibilite import ENTITES
from constants.tag_dp_mapping import BLACK_LIST, TAG_DP_MAPPING
from constants.tokens_eligibilite import TOKENS_ELIGIBILITE
from spacy.matcher import Matcher, PhraseMatcher

# commons
TOKENS_ELIGIBILITE = [token.lower() for token in TOKENS_ELIGIBILITE]

DEBUG_KEYWORD = "france travail"


def debug_targeted(keyword: str, step: str, additional_info: str = ""):
    """
    Fonction de debug ciblée simple

    Args:
        keyword: Mot-clé à rechercher dans additional_info
        step: Étape du processus
        additional_info: Information à afficher (sera vérifiée pour le keyword)
    """
    if keyword.lower() in additional_info.lower():
        print(f"🔍 DEBUG {step}: {additional_info}")


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

            debug_targeted(DEBUG_KEYWORD, "MAPPING", f"'{k}' -> '{lemma_key}'")

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

        debug_targeted(DEBUG_KEYWORD, "MATCHER", f"Pattern ajouté: '{text}'")

        patterns.append(doc)

    phrase_matcher.add("CRITERE_ELIGIBILITE", patterns)

    # Configure the regex matcher for the patterns
    matcher = Matcher(nlp.vocab)
    # ... reste du code inchangé pour les patterns AGE et QF ...
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
    # Récupérer les matchers de base pour l'éligibilité
    phrase_matcher, matcher = create_eligibility_matcher(nlp)

    # Ajouter les patterns spécifiques aux tarifs
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

    return phrase_matcher, matcher


def filter_transport_fare(paragraphs, nlp):
    """Filter paragraphs to keep only those with eligibility criteria"""
    filtered_paragraphs = []
    relevant_sentences = []

    # Créer les matchers une seule fois
    phrase_matcher, matcher = create_transport_fare_matcher(nlp)

    for paragraph in paragraphs:
        doc = nlp(paragraph)

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


def get_matches_and_lemmas(text: str, nlp) -> tuple:
    """Extrait les matches et les lemmes à partir du texte."""

    if "france" in text.lower() or "travail" in text.lower():
        debug_targeted(
            DEBUG_KEYWORD,
            "INPUT",
            "Texte analysé contient 'france' ou 'travail'",
        )

        # Afficher les extraits pertinents
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "france" in line.lower() and "travail" in line.lower():
                debug_targeted(
                    DEBUG_KEYWORD, "INPUT", f"Ligne {i}: {line.strip()}"
                )

    # Créer le matcher
    phrase_matcher, matcher = create_eligibility_matcher(nlp)
    text = text.replace("'", "'")
    doc = nlp(text)

    # DEBUG: Vérifier comment "France Travail" est lemmatisé dans le texte
    for i, token in enumerate(doc):
        if (
            token.text.lower() == "france"
            and i + 1 < len(doc)
            and doc[i + 1].text.lower() == "travail"
        ):
            next_token = doc[i + 1]
            debug_targeted(
                DEBUG_KEYWORD,
                "LEMMA_CHECK",
                f"Dans texte: '{token.text} {next_token.text}' -> "
                f"lemmes: {token.lemma_.lower()}"
                f" {next_token.lemma_.lower()}",
            )

    # Chercher les critères
    matches_phrase = phrase_matcher(doc)

    debug_targeted(
        DEBUG_KEYWORD,
        "MATCHES",
        f"Nombre de matches phrase: {len(matches_phrase)}",
    )

    # DEBUG: Test manuel du PhraseMatcher
    test_doc = nlp("France Travail")
    test_matches = phrase_matcher(test_doc)
    debug_targeted(
        DEBUG_KEYWORD,
        "TEST_MATCH",
        f"Test sur 'France Travail' seul: {len(test_matches)} matches",
    )

    # CORRECTION: Détection d'entités multi-tokens
    matches_entites = False
    doc_text = doc.text

    for entite in ENTITES:
        if entite in doc_text:
            matches_entites = True
            debug_targeted(
                DEBUG_KEYWORD,
                "ENTITE_DETECTION",
                f"Entité trouvée dans texte: '{entite}'",
            )
            break

    debug_targeted(
        DEBUG_KEYWORD, "ENTITE_CHECK", f"matches_entites = {matches_entites}"
    )

    matches = matcher(doc)

    # Utiliser le mapping mis en cache
    tag_dp_mapping_lemmas = get_cached_mapping_lemmas()

    return doc, matches_phrase, matches_entites, matches, tag_dp_mapping_lemmas


def get_highlighted_sentence(doc, start, end, start_char=None, text=None):
    """Trouve et met en surbrillance une partie de phrase."""
    # Trouver la phrase contenant le match
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

    # Si start_char n'est pas fourni, le calculer à partir du span
    if start_char is None:
        start_char = doc[start].idx
        end_char = doc[end - 1].idx + len(doc[end - 1].text)
    else:
        end_char = start_char + len(text)

    # Créer la phrase avec la partie matchée en surbrillance
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

    # Pour les matches de phrases
    for match_id, start, end in matches_phrase:
        span = doc[start:end]
        if not span.text:  # Vérifier que le span n'est pas vide
            continue

        # Vérification du contexte pour les mots sensibles
        context_window = 2  # Nombre de tokens avant/après à vérifier
        start_context = max(0, start - context_window)
        end_context = min(len(doc), end + context_window)
        context_span = doc[start_context:end_context].text.lower()
        context_span = re.sub(r"\s+", " ", context_span).strip()

        if any(black_term in context_span for black_term in BLACK_LIST):
            print(f"⚠️ Span ignoré car dans la liste noire: {context_span}")
            continue

        # Lemmatiser le span (optimisé)
        span_lemmas = [token.lemma_.lower() for token in span]
        span_text = " ".join(span_lemmas)

        # Debug ciblé
        mapping_result = span_text in tag_dp_mapping_lemmas

        debug_targeted(
            DEBUG_KEYWORD,
            "SPAN",
            f"Span '{span.text}' -> lemmatisé '{span_text}' -> "
            f"trouvé: {mapping_result}",
        )

        # Chercher dans le mapping pré-calculé
        if span_text in tag_dp_mapping_lemmas:
            valeur = tag_dp_mapping_lemmas[span_text].get(field)
            if valeur and valeur not in debug_matches:
                valeurs_uniques.add(valeur)
                debug_matches[valeur] = get_highlighted_sentence(
                    doc, span.start, span.end
                )

    # Pour les entités
    if matches_entites:
        doc_text = doc.text

        for entite in ENTITES:
            if entite in doc_text:
                debug_targeted(
                    DEBUG_KEYWORD, "ENTITE", f"Entité trouvée: '{entite}'"
                )

                # Chercher dans TAG_DP_MAPPING directement
                entity_mapping = None
                for k, v in TAG_DP_MAPPING.items():
                    if k and k.lower() == entite.lower():
                        entity_mapping = v
                        break

                is_mapping_found = entity_mapping is not None
                debug_targeted(
                    DEBUG_KEYWORD,
                    "ENTITE",
                    f"Entité '{entite}' -> mapping trouvé: {is_mapping_found}",
                )

                if entity_mapping:
                    valeur = entity_mapping.get(field)
                    if valeur and valeur not in debug_matches:
                        valeurs_uniques.add(valeur)

                        # Trouver les tokens correspondant à l'entité
                        entite_tokens = entite.split()
                        token_start = None
                        token_end = None

                        # Chercher la séquence de tokens dans le document
                        for i in range(len(doc) - len(entite_tokens) + 1):
                            # Vérifier si la séquence correspond
                            match = True
                            for j, entite_token in enumerate(entite_tokens):
                                if (
                                    doc[i + j].text.lower()
                                    != entite_token.lower()
                                ):
                                    match = False
                                    break

                            if match:
                                token_start = i
                                token_end = i + len(entite_tokens)
                                break

                        # Utiliser get_highlighted_sentence
                        if token_start is not None and token_end is not None:
                            debug_matches[valeur] = get_highlighted_sentence(
                                doc, token_start, token_end
                            )
                        else:
                            # Fallback : highlighting simple
                            start_char = doc_text.find(entite)
                            if start_char != -1:
                                before = doc_text[:start_char]
                                after = doc_text[start_char + len(entite) :]
                                highlighted = f"{before}<mark>{entite}</mark>"
                                highlighted += after
                                debug_matches[valeur] = highlighted

    # Pour les matchs spéciaux (AGE et QF)
    if field == "tag":  # Ces matchs spéciaux ne concernent que les tags
        special_tags = {"AGE": "Age", "QF": "Quotient Familial"}
        for match_id, start, end in matches:
            match_type = nlp.vocab.strings[match_id]
            if match_type in special_tags:
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

    # UNE SEULE analyse du texte
    (
        doc,
        matches_phrase,
        matches_entites,
        matches,
        tag_dp_mapping_lemmas,
    ) = get_matches_and_lemmas(text, nlp)

    # Extraire les tags ET les fournisseurs en parallèle
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
