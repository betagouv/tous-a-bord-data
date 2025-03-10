import asyncio
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from services.grist_client import GristDataService


@st.cache_resource
def init_grist_service():
    """Initialise le service Grist."""
    load_dotenv()
    return GristDataService(
        api_key=os.getenv("GRIST_API_KEY"),
        doc_id="jn4Z4deNRbM9MyGBpCK5Jk",
    )


async def fetch_aoms_from_grist():
    """Charge les données des AOM depuis Grist."""
    try:
        grist_service = init_grist_service()
        aoms = await grist_service.get_aoms()
        return pd.DataFrame([aom.model_dump() for aom in aoms])
    except Exception as e:
        st.error(f"Erreur lors du chargement des AOM depuis Grist: {str(e)}")
        return pd.DataFrame()


def get_aoms_data():
    """
    Récupère les données AOM, soit depuis la session si disponibles,
    soit en les chargeant depuis Grist.
    Returns:
        DataFrame: Les données des AOM
    """
    if "aoms_data" not in st.session_state:
        st.session_state.aoms_data = asyncio.run(fetch_aoms_from_grist())
    return st.session_state.aoms_data
