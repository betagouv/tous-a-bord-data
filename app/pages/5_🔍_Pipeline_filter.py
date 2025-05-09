import re

import markdown
import spacy
import spacy.util
import streamlit as st
from bs4 import BeautifulSoup
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span

# Tokens et expressions critères d'éligibilité (seront traités par SpaCy)
TOKENS_ELIGIBILITE = [
    "€",
    "barre bleue",
    "barre rouge",
    "barres bleues",
    "barres rouges",
    "mobilisé action insertion",
    "accompagnant PMR",
    "accompagnateur",
    "actif",
    "adulte salarié",
    "allocataire",
    "Allocation Adulte Handicapé",
    "allocation RSA activité",
    "Allocation Compensatrice Tierce Personne",
    "Allocation Solidarité Spécifique",
    "Allocation Soutien Familial",
    "Allocation Personnalisée Autonomie",
    "allocation vieillesse",
    "allocations journalières",
    "ancien combattant",
    "apprenti",
    "asile",
    "autonome marche",
    "crédit impôt",
    "sur mention",
    "sur-mention",
    "avenir Jeunes",
    "aveugle",
    "ayant-droit",
    "baccalauréat",
    "bénéficiaire",
    "besoin accompagnement",
    "bourse",
    "bourse échelon",
    "boursier",
    "Carte invalidité",
    "carte transport",
    "Carte Mobilité Inclusion",
    "carte mobilité inclusion",
    "carte SNCF",
    "cécité",
    "certificat médical",
    "chômeur",
    "collège",
    "collégien",
    "combattant",
    "Complémentaire Santé Solidaire",
    "condition ressource",
    "contrat aidé",
    "Contrat Adaptation",
    "contrat Engagement Jeune",
    "contrat Unique Insertion",
    "Corps Européen Solidarité",
    "couple",
    "déficient visuel auditif",
    "demandeur emploi",
    "demandeur emploi indemnisé",
    "demandeur emploi longue durée",
    "demandeur emploi non indemnisé",
    "demi-pensionnaire",
    "Demi-pensionnaire Externe",
    "échelon",
    "école Deuxième chance",
    "élève maternelle",
    "élève primaire",
    "élève secondaire",
    "élève interne",
    "régularisation administrative",
    "autonome fiscalement",
    "demandeur asile",
    "domicilié",
    "résident principal",
    "titulaire Contrat",
    "étudiant",
    "euro",
    "exonération",
    "exonéré impôt",
    "famille nombreuse",
    "fauteuil roulant",
    "formation professionnelle",
    "foyer",
    "foyer fiscal",
    "France Travail",
    "guerre",
    "handicap",
    "imposable",
    "inapte",
    "incapacité permanente",
    "indemnisé",
    "indemnité",
    "indemnité journalière",
    "insertion",
    "interne enfant",
    "invalide civil",
    "invalide travail",
    "invalide",
    "invalidité",
    "invalidité temporaire",
    "jeune",
    "junior",
    "lycéen",
    "lycéen boursier",
    "mal voyant",
    "mention cécité",
    "militaire",
    "minima social",
    "minimum vieillesse",
    "Mission Locale",
    "mutilé",
    "non imposable",
    "non imposition",
    "non rémunéré",
    "non voyant",
    "non-éligible",
    "non-imposable",
    "enfant supplémentaire",
    "participation financière",
    "pension invalidité",
    "Plafond ressource",
    "pôle emploi",
    "présentation attestation",
    "Prestation Compensation Handicap",
    "Prime Activité",
    "Programme Compétence",
    "protection temporaire",
    "quotient familial",
    "recherche emploi",
    "réfugié",
    "rémunération mensuelle",
    "rémunéré",
    "ressortissant ukrainien",
    "ressource",
    "retraite",
    "retraité",
    "retrouvé emploi",
    "revenu mensuel",
    "revenu foyer",
    "revenu imposable",
    "RSA majoré",
    "RSA socle",
    "rupture familiale",
    "salarié",
    "sans sur mention",
    "scolaire",
    "scolarisé",
    "sénior",
    "service civique",
    "solidaire",
    "stagiaire",
    "statut debout pénible",
    "tarif",
    "taux handicap",
    "terminale",
    "tout public",
    "travailleur professionnel",
    "veuve",
    "vivre seul",
    "réduction",
]
TOKENS_ELIGIBILITE = [token.lower() for token in TOKENS_ELIGIBILITE]

# Liste des entités spécifiques à identifier
ENTITES = [
    "BA",
    "AAH",
    "ACCES",
    "ACTP",
    "ADA",
    "AEEH",
    "AFPA",
    "AME",
    "ARE",
    "AS",
    "ASA",
    "ASP",
    "ASPA",
    "ASS",
    "C.A.E",
    "C.A.V",
    "CAE",
    "CAF",
    "CASAR",
    "CAV",
    "CCAS",
    "CDAPH",
    "CDDU",
    "CEJ",
    "CFA",
    "CIE",
    "CM2",
    "CMI",
    "CMU",
    "CMU-C",
    "CMUC",
    "CNASEA",
    "CROUS",
    "CSS",
    "CUI",
    "ESAT",
    "MDPH",
    "MSA",
    "ONAC",
    "PACEA",
    "PCH",
    "PDA",
    "PDE",
    "PDIE",
    "PEC",
    "PMR",
    "QF",
    "RFR",
    "RSA",
    "SEGPA",
    "SMIC",
]


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
                [{"TEXT": {"REGEX": r"\d+"}}, {"LOWER": "ans et plus"}],
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
                # Detect : "10 €"
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
                # Detect : "QF inférieur à 1", "QF supérieur à 1",
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
        # Add a custom pipe to recognize specific entities

        @spacy.Language.component("eligibilite_matcher")
        def eligibilite_matcher(doc):
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

        # Add the component to the pipeline
        nlp.add_pipe("eligibilite_matcher", last=True)
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
