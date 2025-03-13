import os
import tempfile

import pandas as pd
import requests
import streamlit as st
from constants.urls import URL_DATASET_AOM_DOC, URL_PASSIM
from services.transport_gouv_client import get_transport_gouv_aom_dataset
from utils.parser_utils import format_column

st.set_page_config(
    page_title="Mise à jour de la BDD",
    page_icon="🔄",
)


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
            dataset_aom = pd.read_excel(
                tmp_path, sheet_name=sheet_names[0], engine="odf"
            )
            dataset_communes = pd.read_excel(
                tmp_path, sheet_name=sheet_names[1], engine="odf"
            )
            dataset_aom.columns = [
                format_column(col) for col in dataset_aom.columns
            ]
            dataset_communes.columns = [
                format_column(col) for col in dataset_communes.columns
            ]
            st.success(
                "File loaded successfully! "
                f"Sheets: {', '.join(sheet_names)}"
            )
            return dataset_aom, dataset_communes
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None


def process_uploaded_file_csv(uploaded_file):
    """
    Traite le fichier téléchargé et retourne un DataFrame.
    Gère le format CSV avec différents séparateurs.
    Args:
        uploaded_file: Le fichier téléchargé via st.file_uploader
    Returns:
        DataFrame pandas ou None en cas d'erreur
    """
    try:
        preview = uploaded_file.read(1024).decode("utf-8")
        st.write("Aperçu du fichier :")
        st.code(preview)
        uploaded_file.seek(0)
        separators = [";", ",", "\t", "|"]
        for sep in separators:
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=sep,
                    encoding="utf-8",
                    on_bad_lines="warn",
                )
                st.success(
                    f"Fichier lu avec succès en utilisant"
                    f"le séparateur '{sep}'"
                )
                return df
            except Exception as e:
                st.warning(
                    f"Tentative avec séparateur '{sep}'" f"échouée: {str(e)}"
                )
                uploaded_file.seek(0)
        st.error("Impossible de lire le fichier avec les séparateurs standard")
        return None
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
        return None


st.markdown("##### Source des données : transport.gouv.fr")
st.markdown(
    """Les données des AOMs (Autorités Organisatrices de la Mobilité) et des
communes proviennent initialement du **CEREMA**
(Centre d'études et d'expertise sur les risques, l'environnement,
la mobilité et l'aménagement) et sont formattées
par **transport.gouv.fr**.
"""
)
with st.expander("En savoir plus sur le données des AOMs"):
    st.markdown(
        f"""
        Pour plus d'informations sur les Autorités Organisatrices de la
        Mobilité (AOM) et la composition de la base de données, consultez
        la [documentation officielle]({URL_DATASET_AOM_DOC}).
        Cette base de données contient :
        - La liste des AOM
        - Leur périmètre géographique
        - Les communes membres
        - Les informations de contact
        """
    )
dataset_aoms = get_transport_gouv_aom_dataset()
# show the url of most recent aoms dataset on transport.gouv
# st.write(dataset_aoms)

df_aom, df_communes = process_transport_gouv_data(dataset_aoms)
if df_aom is not None and df_communes is not None:
    st.write("Aperçu des données :")
    st.dataframe(df_aom.head(5))
    st.dataframe(df_communes.head(5))

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
2. Accédez à la section AOM
3. Cliquez sur le bouton "Exporter"
4. Sélectionnez le format CSV
5. Importez le fichier téléchargé ci-dessous
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

uploaded_file_csv = st.file_uploader(
    "Importez le fichier téléchargé depuis Passim",
    type=["csv"],
)
if uploaded_file_csv is not None:
    passim_df = process_uploaded_file_csv(uploaded_file_csv)
    if passim_df is not None:
        st.write("Aperçu des données :")
        st.dataframe(passim_df.head(5))
