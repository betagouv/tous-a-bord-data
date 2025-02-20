import asyncio
import os

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

# from models.grist_models import AOM, Commune  # Ajout de cette ligne
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

if data_type == "AOM":
    st.subheader("Statistiques des données existantes (Grist)")
    # Création du DataFrame des AOM existantes
    aoms_data = st.session_state.get("aoms_data", asyncio.run(load_aoms()))
    df_existing = pd.DataFrame([aom.model_dump() for aom in aoms_data])
    # Calcul des statistiques
    col1, col2 = st.columns(2)
    with col1:
        st.write("Nombre d'entrées uniques:")
        stats = {
            "N_SIREN_Groupement": df_existing["N_SIREN_Groupement"].nunique(),
            "N_SIREN_AOM": df_existing["N_SIREN_AOM"].nunique(),
            "Nom_de_l_AOM": df_existing["Nom_de_l_AOM"].nunique(),
            "Commune_principale": (
                df_existing["Commune_principale_De_l_AOM"].nunique()
            ),
            "Id_reseau": df_existing["Id_reseau"].nunique(),
            "Population": df_existing["Population_De_l_AOM"].nunique(),
            "Surface": df_existing["Surface_km2_"].nunique(),
        }
        st.write(pd.Series(stats))
    with col2:
        st.write("Nombre de valeurs non-nulles:")
        non_null_stats = {
            "N_SIREN_Groupement": (
                df_existing["N_SIREN_Groupement"].notna().sum()
            ),
            "N_SIREN_AOM": df_existing["N_SIREN_AOM"].notna().sum(),
            "Nom_de_l_AOM": df_existing["Nom_de_l_AOM"].notna().sum(),
            "Commune_principale": df_existing["Commune_principale_De_l_AOM"]
            .notna()
            .sum(),
            "Id_reseau": df_existing["Id_reseau"].notna().sum(),
            "Population": df_existing["Population_De_l_AOM"].notna().sum(),
            "Surface": df_existing["Surface_km2_"].notna().sum(),
        }
        st.write(pd.Series(non_null_stats))
    # Ajout de statistiques descriptives pour Population et Surface
    st.write("\nStatistiques descriptives:")
    desc_stats = df_existing[
        ["Population_De_l_AOM", "Surface_km2_"]
    ].describe()
    desc_stats.columns = ["Population", "Surface (km²)"]
    st.write(desc_stats)
    st.write(f"\nNombre total d'entrées: {len(df_existing)}")


"""
# Ajout du widget de chargement de fichier
uploaded_file = st.file_uploader(
    f"Charger un fichier pour mettre à jour la table {data_type}",
    type=['csv', 'xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        # Lecture du fichier selon son extension
        if uploaded_file.name.endswith('.csv'):
            df_new = pd.read_csv(uploaded_file)
        else:
            df_new = pd.read_excel(uploaded_file)
        # Affichage des premières lignes du fichier chargé
        st.subheader("Aperçu des données à importer")
        st.dataframe(df_new.head())
        # Récupération des colonnes depuis le modèle approprié
        model_class = AOM if data_type == "AOM" else Commune
        expected_columns = list(model_class.__annotations__.keys())
        # Création du tableau de correspondance
        st.subheader("Correspondance des colonnes")
        # Création d'un DataFrame pour la correspondance
        mapping_df = pd.DataFrame({
            'Colonne attendue': expected_columns,
            'Colonne du fichier': [''] * len(expected_columns)
        })
        # Ajout d'une colonne pour la sélection
        file_columns = [''] + list(df_new.columns)  # Ajout d'une option vide
        # Proposition automatique de correspondance basée sur la similarité
        from difflib import get_close_matches
        for idx, expected_col in enumerate(expected_columns):
            matches = get_close_matches(
                expected_col,
                df_new.columns,
                n=1,
                cutoff=0.6
            )
            if matches:
                mapping_df.loc[idx, 'Colonne du fichier'] = matches[0]
        # Création d'un éditeur de tableau
        edited_df = st.data_editor(
            mapping_df,
            column_config={
                "Colonne attendue": st.column_config.TextColumn(
                    "Colonne attendue",
                    disabled=True,
                ),
                "Colonne du fichier": st.column_config.SelectboxColumn(
                    "Colonne du fichier",
                    options=file_columns,
                    required=False
                )
            },
            hide_index=True,
        )
        if st.button("Valider le mapping"):
            # Stockage du mapping dans la session
            st.session_state.column_mapping = dict(zip(
                edited_df['Colonne du fichier'],
                edited_df['Colonne attendue']
            ))
            st.success("Mapping enregistré!")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")

# Chargement des données selon la sélection
if data_type == "AOM":
    if "aoms_data" not in st.session_state:
        st.session_state.aoms_data = asyncio.run(load_aoms())
else:
    if "communes_data" not in st.session_state:
        st.session_state.communes_data = asyncio.run(load_communes())
"""
