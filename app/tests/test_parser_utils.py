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
