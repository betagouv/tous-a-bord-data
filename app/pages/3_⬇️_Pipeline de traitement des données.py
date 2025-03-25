import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from thefuzz import fuzz
from utils.db_utils import get_postgres_cs

st.title("Pipeline de traitement des données")


# Connect to database and load data
def load_data():
    cs = get_postgres_cs()
    engine = create_engine(cs)
    query_passim = """
    SELECT * FROM transport_offers
    """
    df_passim = pd.read_sql(query_passim, engine)
    return df_passim


def format_autorite(nom):
    """Format the authority name by replacing abbreviations"""
    if not nom:  # Check if nom is None or empty
        return nom
    return nom


def fuzzy_match_aom(autorite, df_aoms, seuil=80):
    """
    Find the best match between an authority and AOMs using fuzzy matching
    """
    if not autorite:
        return None, 0
    # Compute the similarity scores for all AOM names
    scores = df_aoms["nom_aom"].apply(
        lambda x: fuzz.ratio(str(autorite).lower(), str(x).lower())
    )
    best_score = max(scores)
    if best_score >= seuil:
        index_best = scores.argmax()
        return df_aoms.iloc[index_best]["n_siren_aom"], best_score
    return None, best_score


df = load_data()

st.write(f"Nombre total de lignes des données initiales : {len(df)}")
st.write("-----> Suppression des lignes avec 'autorite' non renseignée")
df = df[df["autorite"].notna()]
df = df[df["autorite"].str.strip() != ""]
st.write(f"Nombre total de lignes : {len(df)}")

st.write("-----> Sélection des lignes avec 'type_de_transport' pertinent")
df = df[
    (df["type_de_transport"] == "Transport collectif régional")
    | (df["type_de_transport"] == "Transport collectif urbain")
    | (df["type_de_transport"] == "Transport PMR")
]
st.write(f"Nombre total de lignes : {len(df)}")
df = df.reset_index(drop=True)
occurrences_autorite = df["autorite"].value_counts()
st.write(f"Nombre total d'autorités uniques : {len(occurrences_autorite)}")

st.write("Aperçu des offres avec leurs informations AOMS :")
st.dataframe(df)


if st.button("Rechercher les SIREN"):
    df["n_siren_aom"] = None
    df["fuzzy_match_score"] = 0.0

    progress_bar = st.progress(0)
    status_text = st.empty()

    cs = get_postgres_cs()
    engine = create_engine(cs)
    query_aoms = """
    SELECT nom_aom, n_siren_aom, population_aom,
    surface_km_2, nombre_commune_aom
    FROM aoms
    """
    df_aoms = pd.read_sql(query_aoms, engine)

    autorites_uniques = df["autorite"].dropna().unique()
    total_unique = len(autorites_uniques)
    siren_dict = {}
    scores_dict = {}
    nom_aom_dict = {}
    source_siren_dict = {}
    succes = 0

    for idx, autorite in enumerate(autorites_uniques):
        progress = (idx + 1) / total_unique
        progress_bar.progress(progress)
        status_text.text(
            "Traitement fuzzy matching : "
            f"{idx+1}/{total_unique} - {autorite}"
        )
        siren, score = fuzzy_match_aom(autorite, df_aoms, 90)
        if siren is not None:
            siren_dict[autorite] = siren
            scores_dict[autorite] = score
            nom_aom = df_aoms[df_aoms["n_siren_aom"] == siren]["nom_aom"].iloc[
                0
            ]
            nom_aom_dict[autorite] = nom_aom
            source_siren_dict[autorite] = "fuzzy_match"
            succes += 1
        else:
            siren_dict[autorite] = None
            scores_dict[autorite] = score
            nom_aom_dict[autorite] = None
            source_siren_dict[autorite] = None
    df["n_siren_aom"] = df["autorite"].map(siren_dict)
    df["fuzzy_match_score"] = df["autorite"].map(scores_dict)
    df["nom_aom_matched"] = df["autorite"].map(nom_aom_dict)
    df["source_siren"] = df["autorite"].map(source_siren_dict)
    # Display the results with the comparison
    st.write("Résultats du matching :")
    comparison_df = df[
        ["autorite", "nom_aom_matched", "fuzzy_match_score", "source_siren"]
    ].drop_duplicates()
    st.dataframe(comparison_df)

    st.write("Analyse des correspondances avec la table AOMS :")
    # Number of total AOMs in the database
    total_aoms = df_aoms["n_siren_aom"].nunique()
    total_population = df_aoms["population_aom"].sum()
    total_surface = df_aoms["surface_km_2"].sum()
    total_communes = df_aoms["nombre_commune_aom"].sum()

    # Get AOMs with transport offers
    aoms_with_offers = df[
        (df["fuzzy_match_score"] >= 90) & (df["n_siren_aom"].notna())
    ]["n_siren_aom"].unique()

    # Calculate metrics for AOMs with offers
    aoms_avec_offres = len(aoms_with_offers)
    population_avec_offres = df_aoms[
        df_aoms["n_siren_aom"].isin(aoms_with_offers)
    ]["population_aom"].sum()
    surface_avec_offres = df_aoms[
        df_aoms["n_siren_aom"].isin(aoms_with_offers)
    ]["surface_km_2"].sum()
    communes_avec_offres = df_aoms[
        df_aoms["n_siren_aom"].isin(aoms_with_offers)
    ]["nombre_commune_aom"].sum()

    # Calculate coverage rates
    taux_couverture = (
        (aoms_avec_offres / total_aoms) * 100 if total_aoms > 0 else 0
    )
    taux_population = (
        (population_avec_offres / total_population) * 100
        if total_population > 0
        else 0
    )
    taux_surface = (
        (surface_avec_offres / total_surface) * 100 if total_surface > 0 else 0
    )
    taux_communes = (
        (communes_avec_offres / total_communes) * 100
        if total_communes > 0
        else 0
    )

    st.write(f"Nombre total d'AOMs dans la base : {total_aoms}")
    st.write(
        "Nombre d'AOMs ayant au moins une offre de transport : "
        f"{aoms_avec_offres}"
    )
    st.write(f"Taux de couverture des AOMs : {taux_couverture:.2f}%")
    st.write(f"Taux de couverture de la population : {taux_population:.2f}%")
    st.write(f"Taux de couverture de la surface : {taux_surface:.2f}%")
    st.write(f"Taux de couverture des communes : {taux_communes:.2f}%")

    # Save in the database
    try:
        temp_table_name = "temp_transport_offers"
        df_to_save = df[
            ["id", "n_siren_aom", "fuzzy_match_score", "source_siren"]
        ].copy()

        df_to_save.to_sql(
            temp_table_name, engine, if_exists="replace", index=False
        )

        with engine.connect() as connection:
            update_query = text(
                """
                UPDATE transport_offers
                SET n_siren_aom = temp.n_siren_aom,
                    fuzzy_match_score = temp.fuzzy_match_score,
                    source_siren = temp.source_siren
                FROM temp_transport_offers temp
                WHERE transport_offers.id = temp.id
            """
            )
            connection.execute(update_query)
            connection.execute(text(f"DROP TABLE {temp_table_name}"))
            connection.commit()
        st.success("✅ Données sauvegardées dans PostgreSQL!")
    except Exception as e:
        st.error(f"❌ Erreur lors de la sauvegarde dans PostgreSQL: {str(e)}")
    status_text.empty()
