import pandas as pd
import streamlit as st

# from constants.cerema_columns import AOM_MAPPING, COMMUNES_MAPPING
from constants.urls import URL_TRANSPORT_GOUV
from utils.parser_utils import format_column

st.set_page_config(
    page_title="Mise à jour de la BDD",
    page_icon="🔄",
)


def process_uploaded_file(uploaded_file):
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


def process_updaload_transport_gouv_data():
    """Affiche les informations pour télécharger les données AOM depuis
    transport.data.gouv.fr."""
    transport_data_url = URL_TRANSPORT_GOUV
    st.markdown(
        """
    #### Procédure de mise à jour de la base de données
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
    # Proposer un champ pour uploader un fichier
    st.markdown("#### Importer les données téléchargées")
    uploaded_file = st.file_uploader(
        "Importez le fichier téléchargé depuis transport.data.gouv.fr",
        type=["ods"],
    )
    if uploaded_file is not None:
        df = process_uploaded_file(uploaded_file)
        if df is not None:
            # Afficher un aperçu des données
            st.write("Aperçu des données :")
            st.dataframe(df.head(10))
            return df
    return None


# Interface utilisateur
with st.container():
    st.markdown("#### Source des données")
    st.markdown(
        """
    Les données des AOMs (Autorités Organisatrices de la Mobilité) et des
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
    process_updaload_transport_gouv_data()
