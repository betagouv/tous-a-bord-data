import random
import re
import time
import urllib.parse

import pandas as pd
import requests
import streamlit as st
from ratelimit import limits, sleep_and_retry
from sqlalchemy import create_engine
from utils.db_utils import get_postgres_cs

st.title("Pipeline de traitement des données")

# Load data from database


# Connect to database and load data
def load_data():
    cs = get_postgres_cs()
    engine = create_engine(cs)
    df = pd.read_sql("SELECT * FROM transport_offers", engine)
    return df


def extract_siren(text):
    # Search for a SIREN (9 digits)
    siren_pattern = r"\b\d{9}\b"
    matches = re.findall(siren_pattern, text)
    return matches[0] if matches else None


def format_autorite(nom):
    """Format the authority name by replacing abbreviations"""
    if not nom:  # Check if nom is None or empty
        return nom
    replacements = {
        "CA ": "Communauté d'Agglomération ",
        "CC ": "Communauté de Communes ",
    }
    for abbr, full in replacements.items():
        if nom.startswith(abbr):
            return full + nom[len(abbr) :]
    return nom


# Rate limit definition: 7 calls per second maximum
@sleep_and_retry
@limits(calls=6, period=1)  # We set 6 instead of 7 for a safety margin
def call_api(autorite):
    """Call the API with rate limiting management"""
    url = "https://recherche-entreprises.api.gouv.fr/search"
    params = {"q": autorite, "page": 1, "per_page": 1}
    headers = {"accept": "application/json"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 429:
        st.warning(f"Rate limit reached for {autorite}, waiting...")
        time.sleep(1)  # Wait 1 second before retrying
        return call_api(autorite)  # Retry
    response.raise_for_status()
    return response.json()


@sleep_and_retry
@limits(calls=1, period=1)  # Maximum 1 call per second
def search_siren(nom):
    if not nom:  # Check if nom is None or empty
        return None

    def try_api_call(query):
        encoded_query = urllib.parse.quote(query)
        url = (
            "https://recherche-entreprises.api.gouv.fr/search?"
            f"q={encoded_query}&page=1&per_page=1"
        )
        try:
            response = requests.get(url)
            if response.status_code == 429:  # Rate limit atteint
                time.sleep(5)
                return try_api_call(query)  # Retry
            response.raise_for_status()
            data = response.json()
            if data["results"] and len(data["results"]) > 0:
                time.sleep(random.uniform(1, 2))
                return data["results"][0]["siren"]
            return None
        except Exception as e:
            st.error(f"❌ Erreur pour {query}: {str(e)}")
            return None

    siren = try_api_call(nom)
    if siren:
        return siren

    nom_formate = format_autorite(nom)
    if nom_formate != nom:
        siren = try_api_call(nom_formate)
        if siren:
            return siren

    if "SM" in nom.upper():
        query = nom.upper().replace("SM", "Société Mixte")
        siren = try_api_call(query)
        if siren:
            return siren

    if "SI" in nom.upper():
        query = nom.upper().replace("SI", "Syndicat Intercommunal")
        siren = try_api_call(query)
        if siren:
            return siren

    if "CU" in nom.upper():
        query = nom.upper().replace("CU", "Communauté Urbaine")
        siren = try_api_call(query)
        if siren:
            return siren

    if "(" in nom and ")" in nom:
        query = re.sub(r"\([^)]*\)", "", nom).strip()
        siren = try_api_call(query)
        if siren:
            return siren

    st.error(f"❌ Aucun résultat trouvé pour : {nom}")
    return None


df = load_data()

st.write("Sélection des données pertinentes :")
st.write(f"Nombre total de lignes : {len(df)}")
st.write("Suppression des lignes avec 'autorite' non renseignée")
df = df[df["autorite"].notna()]
df = df[df["autorite"].str.strip() != ""]
st.write(f"Nombre total de lignes : {len(df)}")

st.write("Sélection des lignes avec 'type_de_transport' pertinent")
df = df[
    (df["type_de_transport"] == "Transport collectif régional")
    | (df["type_de_transport"] == "Transport collectif urbain")
    | (df["type_de_transport"] == "Transport PMR")
]
st.write(f"Nombre total de lignes : {len(df)}")
df = df.reset_index(drop=True)

st.write("Données chargées de la base PostgreSQL :")
st.dataframe(df)

if st.button("Rechercher les SIREN"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    df["siren_matched_from_recherche_entreprises_api_gouv"] = None
    # Create a unique list of authorities to process
    autorites_uniques = df["autorite"].dropna().unique()
    total_unique = len(autorites_uniques)
    siren_dict = {}
    succes = 0
    for idx, autorite in enumerate(autorites_uniques):
        progress = (idx + 1) / total_unique
        progress_bar.progress(progress)
        status_text.text(f"Traitement : {idx+1}/{total_unique} - {autorite}")
        try:
            siren = search_siren(autorite)
            siren_dict[autorite] = siren
            if siren is not None:
                succes += 1
        except Exception as e:
            st.error(f"❌ Erreur pour {autorite}: {str(e)}")
    taux_succes = (succes / total_unique) * 100
    st.write("Statistiques de recherche :")
    st.write(f"Nombre total d'autorités uniques traitées : {total_unique}")
    st.write(f"Nombre de SIREN trouvés : {succes}")
    st.write(f"Taux de succès : {taux_succes:.2f}%")
    df["siren_matched_from_recherche_entreprises_api_gouv"] = df[
        "autorite"
    ].map(siren_dict)
    st.dataframe(df)

    # try:
    #     cs = get_postgres_cs()
    #     engine = create_engine(cs)
    #     temp_table_name = 'temp_transport_offers'
    #     df = df.reset_index()
    #     df[['id', 'siren_matched_from_recherche_entreprises_api_gouv']]
    # .to_sql(
    #         temp_table_name,
    #         engine,
    #         if_exists='replace',
    #         index=False
    #     )
    #     with engine.connect() as connection:
    #         update_query = text("""
    #             UPDATE transport_offers
    #             SET siren_matched_from_recherche_entreprises_api_gouv =
    #             temp.siren_matched_from_recherche_entreprises_api_gouv
    #             FROM temp_transport_offers temp
    #             WHERE transport_offers.id = temp.id
    #         """)
    #         connection.execute(update_query)
    #         connection.execute(text(f"DROP TABLE {temp_table_name}"))
    #         connection.commit()
    #     st.success("✅ Données sauvegardées dans PostgreSQL!")
    # except Exception as e:
    #     st.error(f"❌ Erreur lors de la sauvegarde dans PostgreSQL: {str(e)}")
    # status_text.empty()
    # st.write("Résultats finaux :")
    # st.dataframe(df)
