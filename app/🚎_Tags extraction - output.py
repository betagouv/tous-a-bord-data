import asyncio
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from services.grist_service import GristDataService
from utils.dataframe_utils import filter_dataframe

# Configuration de la page Streamlit (DOIT ÊTRE EN PREMIER)
st.set_page_config(
    page_title="Base de données des critères d'éligibilité par AOM",
    page_icon="🚌",
    layout="wide",
)

load_dotenv()


async def fetch_aoms_from_grist():
    """Charge les données des AOM depuis Grist."""
    try:
        # Utilisation du singleton pattern
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
            doc_id=os.getenv("GRIST_DOC_INPUTDATA_ID"),
        )
        aoms = await grist_service.get_aoms()
        return pd.DataFrame([aom.model_dump() for aom in aoms])
    except Exception as e:
        st.error(f"Erreur lors du chargement des AOM depuis Grist: {str(e)}")
        return pd.DataFrame()


try:
    if "aoms_data" not in st.session_state:
        st.session_state.aoms_data = asyncio.run(fetch_aoms_from_grist())
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {str(e)}")
    st.session_state.aoms_data = pd.DataFrame()


# Search bar
search_term = st.text_input(
    "🔍 Rechercher dans toutes les colonnes",
    placeholder="Exemple : Bordeaux Métropole",
)

# Filter the data
filtered_df = filter_dataframe(st.session_state.aoms_data, search_term)

# Display the number of results
nb_results = len(filtered_df)
first = f"{nb_results} result{'s' if nb_results > 1 else ''}"
second = f"found{'s' if nb_results > 1 else ''}"

st.write(f"📊{first} {second}")

# Display the filtered table
st.dataframe(
    filtered_df,
    hide_index=True,
)
