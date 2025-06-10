import sys

import pytest

# Ajouter le répertoire app au PYTHONPATH
sys.path.insert(0, "/app")

# flake8: noqa: E402
from services.nlp_services import extract_tags_and_providers, load_spacy_model


# Charger le modèle une seule fois
@pytest.fixture(scope="module")
def nlp():
    """Fixture pour charger le modèle SpaCy une seule fois"""
    return load_spacy_model()


@pytest.mark.parametrize(
    "text,expected_tags,expected_providers",
    [
        # Test France Travail
        (
            "Tarif TRANSITION : dernière notification de droits "
            "délivrée par France Travail",
            ["Demandeur d'emploi"],
            ["FRANCE TRAVAIL"],
        ),
        # Test demandeur d'emploi
        (
            "Tarif réduit pour les demandeurs d'emploi inscrits",
            ["Demandeur d'emploi"],
            ["FRANCE TRAVAIL"],
        ),
        # Test étudiant
        (
            "Tarif étudiant sur présentation de la carte",
            ["Statut étudiant"],
            ["CNOUS"],
        ),
        # Test handicap
        (
            "Tarif réduit pour les personnes en situation de handicap",
            ["Statut handicapé"],
            ["CNSA"],
        ),
        # Test quotient familial
        (
            "Tarif selon quotient familial inférieur à 1200",
            ["Quotient Familial"],
            ["CNAF / MSA"],
        ),
        # Test age pattern
        ("Tarif jeune pour les personnes de 18 à 25 ans", ["Age"], []),
        # Test boursier (cas positif)
        ("bourse échelon 2", ["Statut boursier"], ["CNOUS"]),
        # Test retraité
        ("Tarif senior pour les retraités", ["Retraité"], ["CNAV"]),
    ],
)
def test_detection_positive(nlp, text, expected_tags, expected_providers):
    """
    Test la détection positive d'entités : les cas qui DOIVENT être détectés
    Args:
        nlp: Modèle SpaCy chargé
        text: Texte à analyser
        expected_tags: Tags attendus
        expected_providers: Fournisseurs attendus
    """
    tags, providers, _, _ = extract_tags_and_providers(text, nlp)

    for expected_tag in expected_tags:
        error_msg = (
            f"Tag '{expected_tag}' non trouvé dans '{text}'. "
            f"Tags trouvés: {tags}"
        )
        assert expected_tag in tags, error_msg

    for expected_provider in expected_providers:
        error_msg = (
            f"Fournisseur '{expected_provider}' non trouvé dans '{text}'. "
            f"Fournisseurs trouvés: {providers}"
        )
        assert expected_provider in providers, error_msg


@pytest.mark.parametrize(
    "text,forbidden_tags,forbidden_providers",
    [
        # Test blacklist bourse aux vélos
        (
            "Participation à la bourse aux vélos organisée par la ville",
            ["Statut boursier"],
            ["CNOUS"],
        ),
        # Test blacklist bourse du travail
        (
            "Sélectionnez votre quartier  Bourse",
            ["Statut boursier"],
            ["CNOUS"],
        ),
        # Test blacklist place de la bourse
        (
            "Rendez-vous place de la bourse à 14h",
            ["Statut boursier"],
            ["CNOUS"],
        ),
        # Test blacklist quartier bourse
        (
            "Magasin situé dans le quartier bourse",
            ["Statut boursier"],
            ["CNOUS"],
        ),
        # Test texte neutre
        ("Horaires d'ouverture : lundi au vendredi de 9h à 17h", [], []),
    ],
)
def test_detection_negative(nlp, text, forbidden_tags, forbidden_providers):
    """
    Test la détection négative : les cas qui NE DOIVENT PAS être détectés
    Args:
        nlp: Modèle SpaCy chargé
        text: Texte à analyser
        forbidden_tags: Tags qui ne doivent pas être détectés
        forbidden_providers: Fournisseurs qui ne doivent pas être détectés
    """
    tags, providers, _, _ = extract_tags_and_providers(text, nlp)

    for forbidden_tag in forbidden_tags:
        error_msg = (
            f"Tag '{forbidden_tag}' détecté à tort dans '{text}'. "
            f"Tags trouvés: {tags}"
        )
        assert forbidden_tag not in tags, error_msg

    for forbidden_provider in forbidden_providers:
        error_msg = (
            f"Fournisseur '{forbidden_provider}' détecté à tort dans "
            f"'{text}'. Fournisseurs trouvés: {providers}"
        )
        assert forbidden_provider not in providers, error_msg
