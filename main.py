import asyncio
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from services.grist_service import GristDataService
from utils.dataframe_utils import filter_dataframe

FLY_API_TOKEN = os.getenv("FLY_API_TOKEN")
APP_NAME = "tous-a-bord-data"

load_dotenv()


# utils
async def fetch_aoms_from_grist():
    """Charge les données des AOM depuis Grist."""
    try:
        # Utilisation du singleton pattern
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
        )
        doc_id = os.getenv("GRIST_DOC_OUTPUT_ID")
        aoms = await grist_service.get_aom_tags(doc_id)

        # Filtrer les tags "L" des criteres_eligibilite et fournisseurs
        for aom in aoms:
            if aom.criteres_eligibilite:
                aom.criteres_eligibilite = [
                    tag for tag in aom.criteres_eligibilite if tag != "L"
                ]
            if aom.fournisseurs:
                aom.fournisseurs = [
                    fournisseur
                    for fournisseur in aom.fournisseurs
                    if fournisseur != "L"
                ]

        return pd.DataFrame([aom.model_dump() for aom in aoms])
    except Exception as e:
        st.error(f"Erreur lors du chargement des AOM depuis Grist: {str(e)}")
        return pd.DataFrame()


def launch_fly_machines():
    sirens = st.session_state.aoms_data["siren"].dropna().unique()
    for siren in sirens:
        machine_config = {
            "name": f"batch-job-{siren}",
            "config": {
                "image": "registry.fly.io/tous-a-bord-data:latest",
                "env": {
                    "SIREN": str(siren),
                    "GRIST_API_KEY": os.getenv("GRIST_API_KEY"),
                    "GRIST_DOC_ID": os.getenv("GRIST_DOC_ID"),
                },
                "restart": {"policy": "no"},
                "auto_destroy": True,
                "size": "performance-1x",
                "region": "cdg",
            },
        }
        response = requests.post(
            f"https://api.machines.dev/v1/apps/{APP_NAME}/machines",
            json=machine_config,
            headers={"Authorization": f"Bearer {FLY_API_TOKEN}"},
        )
        if response.status_code != 200:
            st.error(f"Erreur pour {siren}: {response.text}")


# UI
st.set_page_config(
    page_title="Critères d'éligibilité à la tarification sociale et solidaire des transports",
    page_icon="🚌",
    layout="wide",
)

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

# Safely calculate population
try:
    # Check if filtered_df is a DataFrame and has the population_aom column
    if (
        isinstance(filtered_df, pd.DataFrame)
        and "population_aom" in filtered_df.columns
    ):
        # Filter out None values and convert to numeric
        population_values = pd.to_numeric(
            filtered_df["population_aom"].dropna(), errors="coerce"
        )
        population = population_values.sum()
    else:
        # If filtered_df is not a DataFrame or doesn't have the column, set population to 0
        population = 0
except Exception as e:
    st.warning(f"Erreur lors du calcul de la population: {str(e)}")
    population = 0

first = f"**{nb_results} AOM{'s' if nb_results > 1 else ''}**"
second = f"trouvée{'s' if nb_results > 1 else ''}"
taux_population = population * 100 / 66647129 if population > 0 else 0
st.write(
    f"📊{first} {second}, soit " f"**{taux_population:.2f}% de la population**"
)

# Display the filtered table
st.dataframe(
    filtered_df,
    hide_index=True,
)

st.button("Lancer le batch", on_click=launch_fly_machines)
