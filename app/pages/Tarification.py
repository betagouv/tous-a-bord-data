import asyncio
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from services.grist_client import GristDataService


@st.cache_resource
def init_grist_service():
    load_dotenv()
    return GristDataService(
        api_key=os.getenv("GRIST_API_KEY"),
        doc_id="jn4Z4deNRbM9MyGBpCK5Jk",
    )


async def load_aoms():
    """Charge les données des AOM avec leurs informations de tarification."""
    try:
        grist_service = init_grist_service()
        aoms = await grist_service.get_aoms()
        return pd.DataFrame([aom.model_dump() for aom in aoms])
    except Exception as e:
        st.error(f"Erreur lors du chargement des AOM: {str(e)}")
        return pd.DataFrame()


def filter_dataframe(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """Filtre le DataFrame en fonction du terme de recherche."""
    if not search_term:
        return df

    df_str = df.astype(str).apply(lambda x: x.str.lower())
    search_terms = search_term.lower().split()
    mask = pd.Series([True] * len(df), index=df.index)
    for term in search_terms:
        term_mask = pd.Series([False] * len(df), index=df.index)
        for col in df_str.columns:
            term_mask |= df_str[col].str.contains(term, na=False, regex=True)
        mask &= term_mask
    return df[mask]


st.title("Informations de tarification")

# Chargement initial des données
if "aoms_data" not in st.session_state:
    st.session_state.aoms_data = asyncio.run(load_aoms())

# Barre de recherche
search_term = st.text_input(
    "🔍 Rechercher dans toutes les colonnes",
    placeholder="Exemple : Bordeaux Métropole",
)

# Filtrer les données
filtered_df = filter_dataframe(st.session_state.aoms_data, search_term)

# Afficher le nombre de résultats
nb_results = len(filtered_df)
first = f"{nb_results} résultat{'s' if nb_results > 1 else ''}"
second = f"trouvé{'s' if nb_results > 1 else ''}"

st.write(f"📊{first} {second}")

# Afficher le tableau filtré
st.dataframe(
    filtered_df,
    column_config={
        "N_SIREN_AOM": "SIREN AOM",
        "Nom_de_l_AOM": "Nom de l'AOM",
        "Region": "Région",
        "Type_tarification": "Type de tarification",
        "Description_tarification": "Description",
        "Conditions_eligibilite": "Conditions d'éligibilité",
        "Justificatifs": "Justificatifs requis",
        "Prix": st.column_config.NumberColumn(
            "Prix", help="Prix en euros", format="%.2f €"
        ),
    },
    hide_index=True,
)
