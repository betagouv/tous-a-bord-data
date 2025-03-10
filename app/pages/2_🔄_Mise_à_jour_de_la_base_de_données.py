import pandas as pd
import streamlit as st

# from constants.cerema_columns import AOM_MAPPING, COMMUNES_MAPPING
from constants.urls import URL_PASSIM, URL_TRANSPORT_GOUV
from utils.parser_utils import format_column

st.set_page_config(
    page_title="Mise à jour de la BDD",
    page_icon="🔄",
)


def process_uploaded_file_ods(uploaded_file):
    """
    Traite le fichier téléchargé et retourne un DataFrame.
    Gère le format ODS et les onglets multiples.
    Args:
        uploaded_file: Le fichier téléchargé via st.file_uploader
    Returns:
        DataFrame pandas ou None en cas d'erreur
    """
    try:
        # Déterminer le type de fichier et le charger en conséquence
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "ods":
            excel_file = pd.ExcelFile(uploaded_file, engine="odf")
            # Récupérer la liste des onglets
            sheet_names = excel_file.sheet_names
            dataframes = {}
            if len(sheet_names) > 1:
                # S'il y a plusieurs onglets,
                # permettre à l'utilisateur de choisir
                selected_sheet = st.selectbox(
                    "Sélectionnez un onglet:", sheet_names
                )
                if file_extension == "ods":
                    df = pd.read_excel(
                        uploaded_file, sheet_name=selected_sheet, engine="odf"
                    )
                    # Formater les noms de colonnes (minuscules,
                    # sans espaces ni caractères spéciaux)
                    df.columns = [format_column(col) for col in df.columns]
                    # Stocker le DataFrame formaté
                    dataframes[selected_sheet] = df
                else:
                    st.error(
                        "Format de fichier non pris en charge. "
                        "Veuillez télécharger un fichier ODS."
                    )
            else:
                # S'il n'y a qu'un seul onglet, le charger directement
                if file_extension == "ods":
                    df = pd.read_excel(uploaded_file, engine="odf")
                else:
                    df = pd.read_excel(uploaded_file)
        st.success(f"Fichier {file_extension.upper()} chargé avec succès!")
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
        return None


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
        # Lire les premières lignes pour diagnostic
        preview = uploaded_file.read(1024).decode("utf-8")
        st.write("Aperçu du fichier :")
        st.code(preview)
        # Remettre le curseur au début du fichier
        uploaded_file.seek(0)
        # Essayer différents séparateurs
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
                # Remettre le curseur au début pour le prochain essai
                uploaded_file.seek(0)
        # Si aucun séparateur n'a fonctionné
        st.error("Impossible de lire le fichier avec les séparateurs standard")
        return None
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
        return None


def process_upload_transport_gouv_data():
    """Affiche les informations pour télécharger les données AOM depuis
    transport.data.gouv.fr."""
    transport_data_url = URL_TRANSPORT_GOUV
    st.markdown(
        """
    **Guide étape par étape :**
    1. Cliquez sur le lien ci-dessous pour ouvrir le site dans un nouvel onglet
    2. Téléchargez le fichier AOMs le plus récent depuis le site
     (format accepté: ODS)
    3. Revenez sur cette page et importez le fichier téléchargé via le
       sélecteur ci-dessous
    """
    )
    # Ajouter un lien direct vers le site
    st.markdown(
        f"""
        <div style="text-align: center; margin: 20px 0;">
            <a href="{transport_data_url}" target="_blank"
               style="background-color: #4CAF50; color: white;
                      padding: 10px 20px; text-decoration: none;
                      border-radius: 5px; font-weight: bold;">
                Ouvrir transport.data.gouv.fr dans un nouvel onglet
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file_ods = st.file_uploader(
        "Importez le fichier téléchargé depuis transport.data.gouv.fr",
        type=["ods"],
    )
    if uploaded_file_ods is not None:
        df = process_uploaded_file_ods(uploaded_file_ods)
        if df is not None:
            st.write("Aperçu des données :")
            st.dataframe(df.head(10))
            return df
    return None


def process_upload_passim_data():
    """Affiche les informations pour télécharger les données AOM depuis
    Passim."""
    passim_data_url = URL_PASSIM
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
            <a href="{passim_data_url}" target="_blank"
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
            st.dataframe(passim_df.head(10))
            return passim_df
    return None


"""Affiche les informations sur les sources de données."""
st.markdown("##### Source des données : transport.gouv.fr")
st.markdown(
    """Les données des AOMs (Autorités Organisatrices de la Mobilité) et des
communes proviennent initialement du **CEREMA**
(Centre d'études et d'expertise sur les risques, l'environnement,
la mobilité et l'aménagement) et sont formattées
par **transport.gouv.fr**.
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
process_upload_transport_gouv_data()
st.markdown("##### Source des données : Passim")
st.markdown(
    """Les données sur les offres de transport proviennent de l'annuaire
Passim du Cerema.
"""
)
process_upload_passim_data()
