import pytest

from app.utils.parser_utils import format_column


@pytest.mark.parametrize(
    "input_col,expected",
    [
        ("N° SIRENAOM", "n_siren_aom"),
        ("N° SIRENGroupement", "n_siren_groupement"),
        ("Département", "departement"),
        ("Région", "region"),
        ("Plan", "plan"),
        ("Nom de l'AOM", "nom_aom"),
        ("Commune principaleDe l'AOM", "commune_principale_aom"),
        ("Forme juridiqueDe l'AOM", "forme_juridique_aom"),
        ("Bassin de mobilité", "bassin_mobilite"),
        ("Nombre de membreDe l'AOM", "nombre_membre_aom"),
        ("PopulationDe l'AOM", "population_aom"),
        ("Nom de présidentDe l'AOM", "nom_president_aom"),
        ("Adresse du siègeDe l'AOM", "adresse_siege_aom"),
        # Columns with numbers
        ("Surface(km2)", "surface_km_2"),
        ("Bassin de mobilité1", "bassin_mobilite_1"),
        ("Bassin de mobilité2", "bassin_mobilite_2"),
        ("Population totale2019 (Banatic)", "population_totale_2019_banatic"),
        ("Population totale2021(INSEE)", "population_totale_2021_insee"),
        ("LienBanatic", "lien_banatic"),
        ("Lien PageWikipédia", "lien_page_wikipedia"),
        ("ID_reseau", "id_reseau"),
        ("Offres sur le territoire de l'AOM", "offre_territoire_aom"),
        ("Offres sur le territoire de l'AOM_1", "offre_territoire_aom_1"),
        ("Offres sur le territoire de l'AOM_2", "offre_territoire_aom_2"),
        ("Adresse mail", "adresse_mail"),
        ("Nom membre", "nom_membre"),
        ("Siren membre", "siren_membre"),
        ("Nombre membres", "nombre_membre"),
        ("type_d_usagers_-_tous", "type_d_usagers_tous"),
        ("type_d_usagers_-_PMR", "type_d_usagers_pmr"),
        ("type_d_usagers_-_faibles revenus", "type_d_usagers_faibles_revenus"),
        (
            "type_d_usagers_-_recherche d'emplois",
            "type_d_usagers_recherche_d_emplois",
        ),
        ("type_d_usagers_-_soins médicaux", "type_d_usagers_soins_medicaux"),
        ("type_d_usagers_-_personnes âgées", "type_d_usagers_personnes_agees"),
        ("type_d_usagers_-_scolaires", "type_d_usagers_scolaires"),
        ("type_d_usagers_-_touristes", "type_d_usagers_touristes"),
        ("mode_de_transport_-_Autocar", "mode_de_transport_autocar"),
        ("mode_de_transport_-_Avion", "mode_de_transport_avion"),
        ("mode_de_transport_-_Bateau", "mode_de_transport_bateau"),
        ("mode_de_transport_-_Bus", "mode_de_transport_bus"),
        ("mode_de_transport_-_Bus navette", "mode_de_transport_bus_navette"),
        (
            "mode_de_transport_-_Deux-roues motorisés",
            "mode_de_transport_deux_roues_motorises",
        ),
        ("mode_de_transport_-_Bateau", "mode_de_transport_bateau"),
        ("mode_de_transport_-_Funiculaire", "mode_de_transport_funiculaire"),
        ("mode_de_transport_-_Marche", "mode_de_transport_marche"),
        ("mode_de_transport_-_Métro", "mode_de_transport_metro"),
        ("mode_de_transport_-_Motorisé", "mode_de_transport_motorise"),
        ("mode_de_transport_-_Poids lourd", "mode_de_transport_poid_lourd"),
        ("mode_de_transport_-_Autocar", "mode_de_transport_autocar"),
        ("mode_de_transport_-_Taxi", "mode_de_transport_taxi"),
        ("mode_de_transport_-_Téléphérique", "mode_de_transport_telepherique"),
        ("mode_de_transport_-_Train", "mode_de_transport_train"),
        ("mode_de_transport_-_Tramway", "mode_de_transport_tramway"),
        ("mode_de_transport_-_Trottinette", "mode_de_transport_trottinette"),
        ("mode_de_transport_-_Vélo", "mode_de_transport_velo"),
        ("mode_de_transport_-_Voiture", "mode_de_transport_voiture"),
        ("territoire(s)_concerné(s)", "territoires_concernes"),
    ],
)
def test_format_column(input_col, expected):
    """
    Test la fonction format_column avec différents cas d'utilisation.
    Args:
        input_col: Nom de colonne en entrée
        expected: Résultat attendu après formatage
    """
    assert (
        format_column(input_col) == expected
    ), f"'{format_column(input_col)}' != '{expected}'"
