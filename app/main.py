import asyncio
import os

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from services.grist_client import GristDataService

# Configuration de la page Streamlit (DOIT ÊTRE EN PREMIER)
st.set_page_config(
    page_title="Explorateur des AOM",
    page_icon="🚌",
    layout="wide",
)


def get_database_connection():
    conn = psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    register_vector(conn)
    return conn


try:
    conn = get_database_connection()
    st.success("Connexion à la base de données réussie!")
    conn.close()
except Exception as e:
    st.error(f"Erreur de connexion à la base de données: {str(e)}")


@st.cache_resource
def init_grist_service():
    load_dotenv()
    return GristDataService(
        api_key=os.getenv("GRIST_API_KEY"),
        doc_id="jn4Z4deNRbM9MyGBpCK5Jk",
    )


async def load_aoms():
    """Charge et affiche les données des AOM."""
    try:
        grist_service = init_grist_service()
        aoms = await grist_service.get_aoms()

        df = pd.DataFrame([aom.model_dump() for aom in aoms])
        st.dataframe(
            df,
            column_config={
                "N_SIREN_AOM": "SIREN",
                "Nom_de_l_AOM": "Nom",
                "Region": "Région",
                "Population_De_l_AOM": st.column_config.NumberColumn(
                    "Population",
                    help="Population de l'AOM",
                    format="%d",
                ),
                "Surface_km2_": st.column_config.NumberColumn(
                    "Surface (km²)",
                    format="%.2f",
                ),
            },
            hide_index=True,
        )
        return aoms
    except Exception as e:
        st.error(f"Erreur lors du chargement des AOM: {str(e)}")
        return []


async def load_communes():
    """Charge et affiche les données des communes."""
    try:
        grist_service = init_grist_service()
        communes = await grist_service.get_communes()

        df = pd.DataFrame([commune.model_dump() for commune in communes])
        pop_col = "Population_totale_2019_Banatic_"

        st.dataframe(
            df,
            column_config={
                "Nom_membre": "Nom",
                "N_INSEE": "INSEE",
                pop_col: st.column_config.NumberColumn(
                    "Population", format="%d"
                ),
                "Surface_km2_": st.column_config.NumberColumn(
                    "Surface (km²)",
                    format="%.2f",
                ),
                "Nom_de_l_AOM": "AOM",
            },
            hide_index=True,
        )
        return communes
    except Exception as e:
        st.error(f"Erreur lors du chargement des communes: {str(e)}")
        return []


# Interface utilisateur
st.title("Explorateur des Autorités Organisatrices de Mobilité (AOM)")

# Sélecteur de données
data_type = st.radio(
    "Choisir les données à afficher",
    ["AOM", "Communes"],
    horizontal=True,
)

# Chargement des données selon la sélection
if data_type == "AOM":
    if "aoms_data" not in st.session_state:
        st.session_state.aoms_data = asyncio.run(load_aoms())
else:
    if "communes_data" not in st.session_state:
        st.session_state.communes_data = asyncio.run(load_communes())
