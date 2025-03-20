import os

import pandas as pd
import psycopg2
import streamlit as st

# from constants.cerema_columns import AOM_MAPPING, COMMUNES_MAPPING
from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine

# from services.transport_gouv_client import filter_datasets_with_fares
from utils.dataframe_utils import filter_dataframe
from utils.db_utils import check_tables_exist, get_postgres_cs
from utils.grist_utils import get_aoms_data

# Configuration de la page Streamlit (DOIT ÊTRE EN PREMIER)
st.set_page_config(
    page_title=("Base de données des critères d'éligibilité par AOM"),
    page_icon="🚌",
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

engine = create_engine(get_postgres_cs())
check_tables_exist(engine)
try:
    aoms_data = get_aoms_data()
    # Try to load data from the database first
    # aoms_data = load_aoms_data_from_db()
    # if aoms_data is None or aoms_data.empty:
    #     # If no data is found in the database, use Grist as fallback
    #     st.warning(
    #         "Aucune donnée trouvée dans la base de données. Veuillez "
    #         "mettre à jour la base de données via l'onglet "
    #         "'Mise à jour de la base de données'."
    #     )
    #     aoms_data = get_aoms_data()
    # else:
    #     st.success("Données chargées depuis la base de données PostgreSQL.")
    # # Store the data in the session
    # st.session_state.aoms_data = aoms_data
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


# show the datasets with fares
# datasets_with_fares = filter_datasets_with_fares()
# st.write(datasets_with_fares)
