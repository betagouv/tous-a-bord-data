import asyncio
import os
import tempfile

import pandas as pd
import requests
import streamlit as st
from constants.urls import URL_DATASET_AOM_DOC, URL_PASSIM
from dotenv import load_dotenv
from services.grist_service import GristDataService
from services.transport_gouv_client import get_aom_dataset
from utils.parser_utils import format_column

st.set_page_config(
    page_title="Mise à jour de la BDD",
    page_icon="🔄",
    layout="wide",
)

load_dotenv()


def process_transport_gouv_data(dataset_aoms):
    """Downloads and processes ODS file from dataset URL, handling both sheets.

    Args:
        dataset_aoms: Dataset containing the ODS file URL
    Returns:
        Tuple (AOM DataFrame, communes DataFrame) or (None, None) if error
    """
    try:
        url = dataset_aoms.get("url")
        if not url:
            st.error("No URL found in dataset")
            return None, None
        st.info("Downloading file...")
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Download error: {response.status_code}")
            return None, None
        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_f:
            tmp_f.write(response.content)
            tmp_path = tmp_f.name
        try:
            excel_file = pd.ExcelFile(tmp_path, engine="odf")
            sheet_names = excel_file.sheet_names
            if len(sheet_names) < 2:
                st.error("File must contain two sheets (AOM and communes)")
                return None, None
            # Specify dtype for n_insee and code_insee_region columns as string
            dtype_specs = {"n_insee": str, "code_insee_region": str}
            dataset_aom = pd.read_excel(
                tmp_path,
                sheet_name=sheet_names[0],
                engine="odf",
                dtype=dtype_specs,
            )
            dataset_com = pd.read_excel(
                tmp_path,
                sheet_name=sheet_names[1],
                engine="odf",
                dtype=dtype_specs,
            )
            dataset_aom.columns = [
                format_column(col) for col in dataset_aom.columns
            ]
            dataset_com.columns = [
                format_column(col) for col in dataset_com.columns
            ]
            # Ensure specific columns are string type after column formatting
            string_columns = [
                "n_insee",
                "code_insee_region",
                "code_departement",
                "code_region",
            ]

            # Function to ensure columns are string type if they exist
            def ensure_string_columns(df, columns):
                for col in columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
                return df

            # Apply to both dataframes
            dataset_aom = ensure_string_columns(dataset_aom, string_columns)
            dataset_com = ensure_string_columns(dataset_com, string_columns)

            # Additional protection: convert any object columns with numeric-looking strings to explicit string type
            # This prevents PyArrow from trying to convert them to numeric types
            def convert_numeric_strings_to_explicit_strings(df):
                for col in df.select_dtypes(include=["object"]).columns:
                    # Check if column contains any string values that look like numbers
                    if (
                        df[col]
                        .apply(lambda x: isinstance(x, str) and x.isdigit())
                        .any()
                    ):
                        df[col] = df[col].astype(str)
                return df

            dataset_aom = convert_numeric_strings_to_explicit_strings(
                dataset_aom
            )
            dataset_com = convert_numeric_strings_to_explicit_strings(
                dataset_com
            )
            st.success(
                "File loaded successfully! Sheets: "
                f"{', '.join(sheet_names)}"
            )
            # Store dataframes in session state for later use
            st.session_state.aoms_df = dataset_aom
            st.session_state.communes_df = dataset_com
            return dataset_aom, dataset_com
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None


st.header("Données actuellement dans la base")

with st.expander("Voir les données actuelles"):
    try:
        # Utilisation du singleton GristDataService
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
            doc_id=os.getenv("GRIST_DOC_INPUTDATA_ID"),
        )

        # Récupération asynchrone des données
        async def fetch_all_data():
            aoms = await grist_service.get_aoms()
            communes = await grist_service.get_communes()
            transport_offers = await grist_service.get_transport_offers()
            return aoms, communes, transport_offers

        # Exécution de la fonction asynchrone
        aoms, communes, transport_offers = asyncio.run(fetch_all_data())

        # Comptage des enregistrements
        aoms_count = len(aoms)
        communes_count = len(communes)
        passim_count = len(transport_offers)

        # Table AOM
        st.subheader("Table AOMs")
        st.metric("Nombre d'enregistrements", aoms_count)
        if aoms_count > 0:
            aoms_df = pd.DataFrame([aom.model_dump() for aom in aoms])
            st.dataframe(aoms_df)

        # Table Communes
        st.subheader("Table Communes")
        st.metric("Nombre d'enregistrements", communes_count)
        if communes_count > 0:
            communes_df = pd.DataFrame(
                [commune.model_dump() for commune in communes]
            )
            st.dataframe(communes_df)

        # Table Passim (Transport Offers)
        st.subheader("Table Passim")
        st.metric("Nombre d'enregistrements", passim_count)
        if passim_count > 0:
            passim_df = pd.DataFrame(
                [offer.model_dump() for offer in transport_offers]
            )
            st.dataframe(passim_df)
    except Exception as e:
        st.error(
            f"Erreur lors de la récupération des données depuis Grist: "
            f"{str(e)}"
        )


st.header("🔄 Mise à jour de la base de données")
st.markdown("##### Source des données : transport.gouv.fr")
st.markdown(
    """Les données des AOMs (Autorités Organisatrices de la Mobilité) et des
communes proviennent initialement du **CEREMA**
(Centre d'études et d'expertise sur les risques, l'environnement,
la mobilité et l'aménagement) et sont formattées
par **transport.gouv.fr**.

Cette base de données contient :
- La liste des AOM
- Leur périmètre géographique
- Les communes membres
- Les informations de contact
"""
)
with st.expander("En savoir plus sur les données des AOMs"):
    st.markdown(
        f"""
        Pour plus d'informations sur les Autorités Organisatrices de la
        Mobilité (AOM) et la composition de la base de données, consultez
        la [documentation officielle]({URL_DATASET_AOM_DOC}).
        """
    )
dataset_aoms = get_aom_dataset()
# show the url of most recent aoms dataset on transport.gouv
# st.write(dataset_aoms)

# Check if there was an error fetching the dataset
if dataset_aoms and "error" in dataset_aoms:
    st.error(
        f"Erreur lors de la récupération des données: {dataset_aoms['error']}"
    )
    df_aom, df_communes = None, None
else:
    df_aom, df_communes = process_transport_gouv_data(dataset_aoms)
if df_aom is not None and df_communes is not None:
    st.write("Aperçu des données :")
    st.dataframe(df_aom.head(5))
    st.dataframe(df_communes.head(5))

# Section Passim
st.markdown("##### Source des données : Passim")
st.markdown(
    """Les données sur les offres de transport proviennent de l'annuaire
Passim du Cerema.
"""
)
# Passim data is uploaded in the next page
st.markdown(
    """
**Guide étape par étape :**
1. Cliquez sur le lien ci-dessous pour ouvrir le site dans un nouvel onglet
2. Accédez à la section Offre de transport
3. Cliquez sur le bouton "Exporter"
4. Sélectionnez le format CSV
5. Importez le fichier téléchargé dans la table comarquage-offretransport du grist
"""
)
st.markdown(
    f"""
    <div style="text-align: center; margin: 20px 0;">
        <a href="{URL_PASSIM}" target="_blank"
            style="background-color: #4CAF50; color: white;
                    padding: 10px 20px; text-decoration: none;
                    border-radius: 5px; font-weight: bold;">
            Ouvrir Passim dans un nouvel onglet
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.header("Mise à jour des données")

has_aoms_data = "aoms_df" in st.session_state
has_communes_data = "communes_df" in st.session_state
has_passim_data = "passim_df" in st.session_state

if "update_performed" not in st.session_state:
    st.session_state.update_performed = False

st.subheader("Résumé des données disponibles")

col1, col2 = st.columns(2)

with col1:
    if has_aoms_data:
        aoms_count = len(st.session_state.aoms_df)
        st.success(f"✅ Données AOM: {aoms_count} lignes")
        if st.button("🔄 AOM", key="btn_aom"):
            st.write("TODO: Mettre à jour les données AOM")
            # if populate_aom_table(st.session_state.aoms_df):
            #     st.success(
            #         "✅ Données AOM chargées avec succès: "
            #         f"{len(st.session_state.aoms_df)} lignes"
            #     )
            #     st.session_state.update_performed = True
            #     st.rerun()
            # else:
            #     st.error("❌ Échec du chargement des données AOM")
    else:
        st.warning("❌ Données AOM: Non disponibles")

with col2:
    if has_communes_data:
        # Utiliser une variable pour stocker la longueur
        communes_count = len(st.session_state.communes_df)
        msg = f"✅ Données Communes: {communes_count} lignes"
        st.success(msg)
        if st.button("🔄 les Communes", key="btn_communes"):
            st.write("TODO: Mettre à jour les données Communes")
            # if populate_communes_table(st.session_state.communes_df):
            #     st.success(
            #         "✅ Données Communes chargées avec succès: "
            #         f"{len(st.session_state.communes_df)} lignes"
            #     )
            #     st.session_state.update_performed = True
            #     st.rerun()
            # else:
            #     st.error("❌ Échec du chargement des données Communes")
    else:
        st.warning("❌ Données Communes: Non disponibles")

if st.session_state.update_performed:
    st.session_state.update_performed = False
