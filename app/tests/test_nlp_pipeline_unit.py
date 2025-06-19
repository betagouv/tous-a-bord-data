import sys

import pytest

# Ajouter le répertoire app au PYTHONPATH
sys.path.insert(0, "/app")

# flake8: noqa: E402
from services.nlp_services import (
    extract_from_matches,
    extract_tags_and_providers,
    get_matches_and_lemmas,
    load_spacy_model,
)


# Charger le modèle une seule fois
@pytest.fixture(scope="module")
def nlp():
    """Fixture pour charger le modèle SpaCy une seule fois"""
    return load_spacy_model()


def test_get_matches_and_lemmas(nlp):
    """Test si get_matches_and_lemmas détecte correctement les matches et lemmes"""
    # Cas positifs - devrait trouver des matches
    text_positive = "bourse échelon 2"
    doc, matches_phrase, matches_entites, matches, _ = get_matches_and_lemmas(
        text_positive, nlp
    )
    assert (
        len(matches_phrase) > 0
    ), "Devrait trouver au moins un match pour 'bourse'"

    text_positive = "place de la bourse"
    doc, matches_phrase, matches_entites, matches, _ = get_matches_and_lemmas(
        text_positive, nlp
    )
    assert (
        len(matches_phrase) > 0
    ), "Devrait trouver au moins un match pour 'bourse'"

    # Cas négatif - ne devrait pas trouver de matches dans un texte neutre
    text_neutral = "Horaires d'ouverture : lundi au vendredi de 9h à 17h"
    doc, matches_phrase, matches_entites, matches, _ = get_matches_and_lemmas(
        text_neutral, nlp
    )
    assert (
        len(matches_phrase) == 0
    ), "Ne devrait pas trouver de matches dans un texte neutre"


def test_blacklist_context_check(nlp):
    """Test si la vérification du contexte fonctionne correctement pour les termes blacklistés"""
    # Cas où le terme est dans la blacklist
    text_blacklisted = "Rendez-vous place de la bourse à 14h"
    (
        doc,
        matches_phrase,
        matches_entites,
        matches,
        tag_dp_mapping_lemmas,
    ) = get_matches_and_lemmas(text_blacklisted, nlp)

    # Simuler un match pour "bourse" dans ce texte
    # Trouver l'index du token "bourse"
    bourse_index = None
    for i, token in enumerate(doc):
        if token.text.lower() == "bourse":
            bourse_index = i
            break

    assert (
        bourse_index is not None
    ), "Le token 'bourse' devrait être trouvé dans le texte"

    # Créer un match simulé
    match_id = nlp.vocab.strings["CRITERE_ELIGIBILITE"]
    simulated_matches_phrase = [(match_id, bourse_index, bourse_index + 1)]

    # Extraire les tags avec ce match simulé
    tags_uniques, _ = extract_from_matches(
        doc,
        simulated_matches_phrase,
        False,
        [],
        tag_dp_mapping_lemmas,
        nlp,
        "tag",
    )

    assert (
        "Statut boursier" not in tags_uniques
    ), "Le tag 'Statut boursier' ne devrait pas être détecté car 'bourse' est dans un contexte blacklisté"


def test_extract_tags_providers_with_blacklist(nlp):
    """Test si extract_tags_and_providers filtre correctement les termes blacklistés"""
    # Tester tous les cas de la blacklist
    blacklist_cases = [
        "Participation à la bourse aux vélos organisée par la ville",
        "Sélectionnez votre quartier Bourse",
        "Rendez-vous place de la bourse à 14h",
        "Magasin situé dans le quartier bourse",
    ]

    for text in blacklist_cases:
        tags, providers, _, _ = extract_tags_and_providers(text, nlp)
        assert (
            "Statut boursier" not in tags
        ), f"'Statut boursier' ne devrait pas être détecté dans '{text}'"
        assert (
            "CNOUS" not in providers
        ), f"'CNOUS' ne devrait pas être détecté dans '{text}'"
