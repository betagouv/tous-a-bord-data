import asyncio
import io
import os

import pandas as pd
import psycopg2
import requests
import streamlit as st
from constants.cerema_columns import AOM_MAPPING, COMMUNES_MAPPING
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


# Récupérer les données depuis le cerema
url_donnees_cerema = (
    "https://www.cerema.fr/fr/system/files?"
    "file=documents/2023/04/base_rt_diffusion.ods"
)


def load_cerema_data():
    """Télécharge et formate les données CEREMA."""
    url = url_donnees_cerema
    try:
        # Télécharger le fichier
        response = requests.get(url)
        excel_file = io.BytesIO(response.content)
        aom_cerema = pd.read_excel(excel_file, engine="odf", sheet_name=0)
        communes_cerema = pd.read_excel(excel_file, engine="odf", sheet_name=1)
        # formatter le nom des variables
        aom_cerema = aom_cerema.rename(columns=AOM_MAPPING)
        communes_cerema = communes_cerema.rename(columns=COMMUNES_MAPPING)
        return aom_cerema, communes_cerema
    except Exception as e:
        st.error(f"Erreur lors du chargement des données CEREMA: {str(e)}")
        return None


# Interface utilisateur


st.title(
    "Moteur d'éligibilité aux tarifs sociaux" " et solidaires de la mobilité"
)

st.header("Source des données :")
st.subheader("CEREMA")


with st.container():
    st.markdown(
        """
    Les données des AOMs hors régions et des communes proviennent du CEREMA
    (Centre d'études et d'expertise sur les risques, l'environnement,
    la mobilité et l'aménagement).
    """
    )
    with st.expander("En savoir plus sur les AOM"):
        st.markdown(
            """
        Une Autorité Organisatrice de la Mobilité (AOM) est l'acteur public
        compétent pour l'organisation de la mobilité sur son territoire.
        Elle a pour mission :
        - L'organisation des services de transport public
        - La gestion des services de mobilité active
        - La contribution aux objectifs de lutte contre le changement
        climatique
        - Le développement des pratiques de mobilité durables et solidaires
        """
        )
    with st.expander("Source des données"):
        st.markdown(
            """
        📊 [Base de données des Autorités Organisatrices de la Mobilité (AOM)]
        (https://www.cerema.fr/fr/actualites/liste-composition-autorites-organisatrices-mobilite-au-1er-4)
        """
        )
    st.download_button(
        label="📥 Télécharger les données brutes",
        data=url_donnees_cerema,
        file_name="base_rt_diffusion.ods",
        mime="application/vnd.oasis.opendocument.spreadsheet",
        help="Télécharger la base de données CEREMA au format ODS",
    )
    aom_cerema = load_cerema_data()
    if aom_cerema is not None:
        st.dataframe(
            aom_cerema,
            column_config={
                "SIREN": st.column_config.TextColumn("SIREN"),
                "Nom": st.column_config.TextColumn("Nom de l'AOM"),
                "Population": st.column_config.NumberColumn(
                    "Population", help="Population de l'AOM", format="%d"
                ),
                "Surface": st.column_config.NumberColumn(
                    "Surface (km²)", format="%.2f"
                ),
            },
            hide_index=True,
        )

st.subheader("France Mobilité")
with st.container():
    st.markdown(
        """
    Les données des AOMs régionales proviennent de France Mobilité
    """
    )
    url = (
        "https://www.francemobilites.fr/outils/"
        "observatoire-politiques-locales-mobilite/aom"
    )
    # ajouter le champ de recherche "région" et exporter

# Récupération des données du GRIST
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
