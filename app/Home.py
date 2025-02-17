import asyncio
import os

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

st.title("Explorateur des Autorités Organisatrices de Mobilité (AOM)")


@st.cache_resource
def init_grist_service():
    load_dotenv()
    return GristDataService(
        api_key=os.getenv("GRIST_API_KEY"),
        doc_id="jn4Z4deNRbM9",
    )


async def load_data():
    try:
        grist_service = init_grist_service()
        aoms = await grist_service.get_aoms()
        return aoms
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return []


if "aoms_data" not in st.session_state:
    try:
        st.session_state.aoms_data = asyncio.run(load_data())
    except Exception as e:
        msg = f"Erreur lors de l'initialisation des données: {str(e)}"
        st.error(msg)
        st.session_state.aoms_data = []
