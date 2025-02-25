import io
import os
import time  # Ajout de l'import time

import pandas as pd
import psycopg2
import requests
import streamlit as st
from constants.cerema_columns import AOM_MAPPING, COMMUNES_MAPPING
from pgvector.psycopg2 import register_vector

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


# Récupérer les données depuis le cerema
url_donnees_cerema = (
    "https://www.cerema.fr/fr/system/files?"
    "file=documents/2023/04/base_rt_diffusion.ods"
)


def load_cerema_data():
    """Télécharge et formate les données CEREMA."""
    url = url_donnees_cerema
    try:
        progress_bar = st.progress(0, "Téléchargement du fichier en cours...")
        # Télécharger le fichier
        response = requests.get(url)
        excel_file = io.BytesIO(response.content)
        progress_bar.progress(30, "Chargement des données des AOM...")
        # Charger et formater les données AOM
        aom_cerema = pd.read_excel(excel_file, engine="odf", sheet_name=0)
        aom_cerema = aom_cerema.rename(columns=AOM_MAPPING)
        aom_cerema = aom_cerema.iloc[:, :-1]
        progress_bar.progress(60, "Chargement des données des communes...")
        # Charger et formater les données Communes
        communes_cerema = pd.read_excel(excel_file, engine="odf", sheet_name=1)
        colonnes_originales = communes_cerema.columns
        COMMUNES_MAPPING_FIXED = {}
        for col in colonnes_originales:
            if "Nom de" in col and "AOM" in col:
                COMMUNES_MAPPING_FIXED[col] = "Nom_de_l_AOM"
            elif "Forme juridique" in col and "AOM" in col:
                COMMUNES_MAPPING_FIXED[col] = "Forme_juridique_De_l_AOM"
            elif col in COMMUNES_MAPPING:
                COMMUNES_MAPPING_FIXED[col] = COMMUNES_MAPPING[col]
        communes_cerema = communes_cerema.rename(
            columns=COMMUNES_MAPPING_FIXED
        )
        communes_cerema = communes_cerema.replace("#RÉF !", None)
        progress_bar.progress(100, "Chargement terminé !")
        time.sleep(1)
        progress_bar.empty()
        return aom_cerema, communes_cerema
    except Exception as e:
        st.error(f"Erreur lors du chargement des données CEREMA: {str(e)}")
        return None, None


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
    aom_cerema, communes_cerema = load_cerema_data()
    if aom_cerema is not None:
        st.success("Données chargées avec succès!")
        st.write("Données des AOM :")
        st.dataframe(aom_cerema, hide_index=True)
        # st.dataframe(aom_cerema.head())
        st.write("Données des communes :")
        st.dataframe(communes_cerema, hide_index=True)
        # st.dataframe(communes_cerema.head())
    else:
        st.error("Erreur lors du chargement des données")
    with st.expander("Source des données"):
        st.markdown(
            """
        📊 [Base de données des Autorités Organisatrices de la Mobilité (AOM)]
        (https://www.cerema.fr/fr/actualites/liste-composition-autorites-organisatrices-mobilite-au-1er-4)
        """
        )

# st.subheader("France Mobilité")
# with st.container():
#     st.markdown(
#         """
#     Les données des AOMs régionales proviennent de France Mobilité
#     """
#     )
#     url = (
#         "https://www.francemobilites.fr/outils/"
#         "observatoire-politiques-locales-mobilite/aom"
#     )
#     # ajouter le champ de recherche "région" et exporter
