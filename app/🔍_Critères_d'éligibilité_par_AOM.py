import os

import psycopg2
import streamlit as st

# from constants.cerema_columns import AOM_MAPPING, COMMUNES_MAPPING
from pgvector.psycopg2 import register_vector
from utils.dataframe_utils import filter_dataframe
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

# Chargement initial des données
aoms_data = get_aoms_data()


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
