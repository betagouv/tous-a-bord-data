import asyncio
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyexcel_ods3
import requests
import streamlit as st
from dotenv import load_dotenv

from constants.urls import URL_PASSIM_AOMS, URL_PASSIM_OFFRES_TRANSPORT
from models.grist_models import Aom, AomTransportOffer, DownloadedAom
from services.grist_service import GristDataService
from services.transport_gouv_client import get_aom_dataset
from utils.parser_utils import format_column

# Page configuration
st.set_page_config(
    page_title="Update input data",
    page_icon="🔄",
    layout="wide",
)

# Load environment variables
load_dotenv()

# Initialize session state
if "update_performed" not in st.session_state:
    st.session_state.update_performed = False
if "aoms_data" not in st.session_state:
    st.session_state.aoms_data = []


def download_file(
    url: str, status_placehorlder, progressbar_placeholder
) -> Optional[str]:
    """
    Download a file with progress tracking

    Args:
        url: URL to download
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        Path to downloaded file or None if error
    """
    status_placehorlder.info("Téléchargement du fichier en cours...")

    # Use a session for better control
    session = requests.Session()

    # First make a HEAD request to get the content length
    head_response = session.head(url)
    total_size = int(head_response.headers.get("content-length", 0))

    # Now make the actual GET request with streaming
    response = session.get(url, stream=True)
    if response.status_code != 200:
        status_placehorlder.error(
            f"Erreur de téléchargement: {response.status_code}"
        )
        return None

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_f:
        # Download with progress tracking
        downloaded = 0
        progress_bar = progressbar_placeholder.progress(0)

        for chunk in response.iter_content(chunk_size=4096):
            if chunk:  # filter out keep-alive chunks
                tmp_f.write(chunk)
                downloaded += len(chunk)

                # Update progress bar
                if total_size > 0.0:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress)

        # Download complete
        progress_bar.progress(1.0)
        return tmp_f.name


def process_ods_file(
    file_path: str, status_placeholder, progressbar_placeholder
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process ODS file directly using pyexcel_ods3

    Args:
        file_path: Path to the ODS file
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        Tuple of (AOM data list)
    """
    # Create a processing progress tracker
    status_placeholder.info("Traitement du fichier ODS...")
    processing_progress = progressbar_placeholder.progress(0)

    try:
        # Step 1: Load the ODS file directly with pyexcel_ods3 (much faster than pandas)
        processing_progress.progress(0.1)

        # Load the ODS file directly
        data = pyexcel_ods3.get_data(file_path)
        # Check if we have at least 2 sheets
        if len(data) < 2:
            st.error(
                "Le fichier doit contenir deux feuilles (AOM et communes)"
            )
            return [], []

        # Get sheet names (keys of the data dictionary)
        sheet_names = list(data.keys())

        # Step 2: Process AOM data directly
        status_placeholder.info("Traitement des données AOM...")
        processing_progress.progress(0.3)

        # Get AOM data
        aom_sheet = data[sheet_names[0]]

        # Extract headers and format them
        aom_headers = [format_column(header) for header in aom_sheet[0]]

        # Process AOM records directly
        aom_data = []
        for row in aom_sheet[1:]:  # Skip header row
            # Create a dictionary from headers and row values
            row_dict = {}
            for j, header in enumerate(aom_headers):
                if j < len(row):
                    row_dict[header] = row[j]
                else:
                    row_dict[header] = None

            # Store the raw data
            aom_data.append(row_dict)

        # Complete
        processing_progress.progress(1.0)

        return aom_data
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier: {str(e)}")
        return [], []


def validate_data(
    raw_data: List[Dict], status_placeholder, progress_placeholder
) -> List[Aom]:
    """
    Validate raw AOM data with Pydantic using intermediate model

    Args:
        raw_data: List of dictionaries containing AOM data
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        List of validated Aom objects
    """
    status_placeholder.info(
        f"Validation de {len(raw_data)} enregistrements avec Pydantic..."
    )
    validated_data = []
    errors = 0

    progress_bar = progress_placeholder.progress(0)

    batch_size = 100
    for i in range(0, len(raw_data), batch_size):
        batch = raw_data[i : i + batch_size]

        # Process each item in batch
        for idx, row_dict in enumerate(batch):
            try:
                # First validate with the intermediate model
                downloaded_aom = DownloadedAom.model_validate(row_dict)
                aom = downloaded_aom.to_aom()
                validated_data.append(aom)

            except Exception as e:
                # Capture errors lors de la validation DownloadedAom, or when convert to Aom model
                record_num = i + idx + 1
                error_message = str(e)
                status_placeholder.error(
                    f"Erreur de validation DownloadedAom pour l'enregistrement {record_num}: {error_message}"
                )

                errors += 1

        # Update progress after each batch
        progress = min((i + batch_size) / len(raw_data), 1.0)
        progress_bar.progress(progress)

    # Complete progress
    progress_bar.progress(1.0)

    status_placeholder.success(
        f"Validation terminée: {len(validated_data)} objets valides, {errors} erreur{'s' if errors > 0 else ''}"
    )

    progress_bar.empty()

    return validated_data


async def update_aoms_in_grist(
    aoms: List[Aom], status_placeholder, progressbar_placeholder
) -> bool:
    """
    Update AOM data in Grist

    Args:
        aoms: List of Aom objects
        status_placeholder: Streamlit placeholder for status updates
        progressbar_placeholder: Streamlit placeholder for progress bar

    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Get GristDataService instance
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
        )

        # Get document ID
        input_data_doc_id = os.getenv("GRIST_DOC_INPUT_ID")

        # First, delete any records that are no longer needed
        status_placeholder.info("Suppression des enregistrements obsolètes...")
        try:
            delete_result = await grist_service.delete_aoms(
                aoms, doc_id=input_data_doc_id
            )
            if delete_result.get("deleted", 0) > 0:
                status_placeholder.info(
                    f"{delete_result.get('deleted')} enregistrements obsolètes supprimés"
                )
        except Exception as e:
            error_msg = f"Exception lors de la suppression des enregistrements obsolètes: {str(e)}"
            logging.exception(error_msg)
            status_placeholder.warning(error_msg)
            # Continue with the update even if deletion fails

        # Create progress tracking
        status_placeholder.info("Mise à jour des données dans Grist...")
        progress_bar = progressbar_placeholder.progress(0)

        # Define batch size
        batch_size = 50  # Optimal batch size for performance

        # Create batches
        batches = [
            aoms[i : i + batch_size] for i in range(0, len(aoms), batch_size)
        ]
        total_batches = len(batches)

        # Process each batch
        total_updated = 0
        for i, batch in enumerate(batches):
            status_placeholder.info(
                f"Traitement du lot {i+1}/{total_batches} ({len(batch)} aoms par batch)"
            )

            try:
                # Update AOM data for this batch
                await grist_service.update_aoms(
                    batch, doc_id=input_data_doc_id
                )

                # Always consider the update successful since we're getting a 200 response
                # The actual response is displayed with st.info in the UI
                total_updated += len(batch)

                # Update progress
                progress = min((i + 1) / total_batches, 1.0)
                progress_bar.progress(progress)
            except Exception as e:
                error_msg = f"Exception lors de la mise à jour du lot {i+1}/{total_batches}: {str(e)}"
                logging.exception(error_msg)
                status_placeholder.error(error_msg)
                return False

        # Complete progress
        progress_bar.progress(1.0)
        status_placeholder.success(
            f"Total des enregistrements mis à jour: {total_updated}"
        )
        progress_bar.empty()

        # Set update flag
        st.session_state.update_performed = True

        return True
    except Exception as e:
        status_placeholder.error(
            f"Erreur lors de la mise à jour des données AOM: {str(e)}"
        )
        return False


async def fetch_current_data():
    """
    Fetch current data from Grist

    Returns:
        couple of (aoms, transport_offers)
    """
    try:
        # Get GristDataService instance
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
        )

        # Fetch data
        input_data_doc_id = os.getenv("GRIST_DOC_INPUT_ID")
        aoms = await grist_service.get_aoms(doc_id=input_data_doc_id)
        comarquage_transport_offers = (
            await grist_service.get_comarquage_transport_offers(
                doc_id=input_data_doc_id
            )
        )
        comarquage_aoms = await grist_service.get_comarquage_aoms(
            doc_id=input_data_doc_id
        )

        return aoms, comarquage_transport_offers, comarquage_aoms
    except Exception as e:
        st.error(
            f"Erreur lors de la récupération des données depuis Grist: {str(e)}"
        )
        return [], [], []


def download_and_process_aom_data(
    dataset_info, status_placeholder, progressbar_placeholder
):
    """
    Download and process AOM data

    Args:
        dataset_info: Dataset information containing URL
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        True if successful, False otherwise
    """
    # Get URL
    url = dataset_info.get("url")
    if not url:
        status_placeholder.error(
            "URL non trouvée dans les informations du dataset"
        )
        return False

    # Download file
    file_path = download_file(url, status_placeholder, progressbar_placeholder)
    if not file_path:
        return False

    try:
        # Process file
        aom_data = process_ods_file(
            file_path, status_placeholder, progressbar_placeholder
        )

        # Check if data was processed successfully
        if not aom_data:
            status_placeholder.error("Erreur lors du traitement des données")
            return False

        # Filtrer les lignes vides avant validation
        filtered_aom_data = []
        empty_rows = 0
        for row_dict in aom_data:
            if any(value is not None for value in row_dict.values()):
                filtered_aom_data.append(row_dict)
            else:
                empty_rows += 1

        if empty_rows > 0:
            status_placeholder.info(
                f"{empty_rows} lignes vides ont été ignorées."
            )

        # Validate AOM data with Pydantic immediately
        status_placeholder.info(
            f"Validation de {len(filtered_aom_data)} enregistrements avec Pydantic..."
        )
        validated_aoms = validate_data(
            filtered_aom_data, status_placeholder, progressbar_placeholder
        )

        # Store validated objects in session state
        st.session_state.aoms_data = validated_aoms

        return True
    except Exception as e:
        st.error(f"Erreur lors du traitement des données: {str(e)}")
        return False
    finally:
        # Clean up temporary file
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)


def validate_aom_transport_offers(
    data_df, status_placeholder, progress_placeholder
):
    """
    Validate AOM transport offers data with Pydantic

    Args:
        data_df: DataFrame containing AOMs with transport offers
        status_placeholder: Streamlit placeholder for status updates
        progress_placeholder: Streamlit placeholder for progress bar

    Returns:
        List of validated AomTransportOffer objects
    """
    status_placeholder.info(
        f"Validation de {len(data_df)} enregistrements avec Pydantic..."
    )
    validated_data = []
    errors = 0

    progress_bar = progress_placeholder.progress(0)

    # Process each row in the DataFrame
    for idx, row in data_df.iterrows():
        try:
            # Convert row to dict and validate with AomTransportOffer model
            row_dict = row.to_dict()
            aom_offer = AomTransportOffer.model_validate(row_dict)
            validated_data.append(aom_offer)
        except Exception as e:
            # Capture validation errors
            record_num = idx + 1
            error_message = str(e)
            status_placeholder.error(
                f"Erreur de validation pour l'enregistrement {record_num}: {error_message}"
            )
            errors += 1

        # Update progress
        progress = min((idx + 1) / len(data_df), 1.0)
        progress_bar.progress(progress)

    # Complete progress
    progress_bar.progress(1.0)

    status_placeholder.success(
        f"Validation terminée: {len(validated_data)} objets valides, {errors} erreur{'s' if errors > 0 else ''}"
    )

    progress_bar.empty()

    return validated_data


async def upload_aoms_with_offers(
    data_df, status_placeholder, progressbar_placeholder
):
    """
    Upload AOMs with transport offers to Grist

    Args:
        data_df: DataFrame containing AOMs with transport offers
        status_placeholder: Streamlit placeholder for status updates
        progressbar_placeholder: Streamlit placeholder for progress bar

    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Get GristDataService instance
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
        )

        # Get document ID
        intermediary_doc_id = os.getenv("GRIST_DOC_INTERMEDIARY_ID")

        # Create progress tracking
        status_placeholder.info("Validation des données...")

        # Validate data with Pydantic
        validated_offers = validate_aom_transport_offers(
            data_df, status_placeholder, progressbar_placeholder
        )

        if not validated_offers:
            status_placeholder.error("Aucune donnée valide à mettre à jour")
            return False

        # First, delete any records that are no longer needed
        status_placeholder.info("Suppression des enregistrements obsolètes...")
        try:
            delete_result = await grist_service.delete_aom_transport_offers(
                validated_offers, doc_id=intermediary_doc_id
            )
            if delete_result.get("deleted", 0) > 0:
                status_placeholder.info(
                    f"{delete_result.get('deleted')} enregistrements obsolètes supprimés"
                )
        except Exception as e:
            error_msg = f"Exception lors de la suppression des enregistrements obsolètes: {str(e)}"
            logging.exception(error_msg)
            status_placeholder.warning(error_msg)
            # Continue with the update even if deletion fails

        status_placeholder.info(
            f"Mise à jour de {len(validated_offers)} offres de transport dans Grist..."
        )
        progress_bar = progressbar_placeholder.progress(0)

        # Define batch size for better performance
        batch_size = 50

        # Create batches
        batches = [
            validated_offers[i : i + batch_size]
            for i in range(0, len(validated_offers), batch_size)
        ]
        total_batches = len(batches)

        # Process each batch
        total_updated = 0
        for i, batch in enumerate(batches):
            status_placeholder.info(
                f"Traitement du lot {i+1}/{total_batches} ({len(batch)} offres par batch)"
            )

            try:
                # Add AOM transport offers for this batch
                await grist_service.update_aom_transport_offers(
                    batch, doc_id=intermediary_doc_id
                )

                # Count updated records
                total_updated += len(batch)

                # Update progress
                progress = min((i + 1) / total_batches, 1.0)
                progress_bar.progress(progress)
            except Exception as e:
                error_msg = f"Exception lors de la mise à jour du lot {i+1}/{total_batches}: {str(e)}"
                logging.exception(error_msg)
                status_placeholder.error(error_msg)
                return False

        # Complete progress
        progress_bar.progress(1.0)
        status_placeholder.success(
            f"Total des enregistrements mis à jour: {total_updated}"
        )
        progress_bar.empty()

        return True
    except Exception as e:
        status_placeholder.error(
            f"Erreur lors de la mise à jour des données: {str(e)}"
        )
        return False


# Main UI
st.header("🔄 Update input data")
st.markdown(
    """
    - Cette page permet de mettre à jour les données des AOMs (Autorités Organisatrices de la Mobilité),
    et des offres de transport.
    - Les données sont récupérées depuis le site **transport.gouv.fr** et l'annuaire **Passim** du Cerema.
    """
)

# Current data section
with st.expander(
    "👀 Task 1 : visualisation des données d'entrée actuellement disponibles dans Grist",
    expanded=False,
):
    # Fetch current data
    aoms, comarquage_transport_offers, comarquage_aoms = asyncio.run(
        fetch_current_data()
    )

    # Display data tables
    if aoms:
        st.metric("AOMs source: transport.data.gouv", len(aoms))
        aoms_df = pd.DataFrame([aom.model_dump() for aom in aoms])
        st.dataframe(aoms_df)

    if comarquage_transport_offers:
        st.metric(
            "Offres de transport source: Passim / Comarquage",
            len(comarquage_transport_offers),
        )
        offers_df = pd.DataFrame(
            [offer.model_dump() for offer in comarquage_transport_offers]
        )
        st.dataframe(offers_df)

    if comarquage_aoms:
        st.metric("AOMs source: Passim / Comarquage", len(comarquage_aoms))
        comarquage_aoms_df = pd.DataFrame(
            [aom.model_dump() for aom in comarquage_aoms]
        )
        st.dataframe(comarquage_aoms_df)

with st.expander(
    "⬇️ Task 2 : download de la dernière version des données AOMs depuis **transport.data.gouv**",
    expanded=False,
):
    st.markdown(
        """
        Les données des AOMs (Autorités Organisatrices de la Mobilité)
        proviennent du **CEREMA** et sont formattées par **transport.gouv.fr**.
    """
    )
    # Get dataset info
    dataset_aoms = get_aom_dataset()

    # Download section
    download_container = st.container()
    with download_container:
        # Check if there was an error fetching the dataset info
        if dataset_aoms and "error" in dataset_aoms:
            st.error(
                f"Erreur API lors de la récupération des informations sur le dataset: {dataset_aoms['error']}"
            )
        elif dataset_aoms:
            # Show dataset info
            st.info(
                f"Dataset disponible: {dataset_aoms.get('title', 'Données AOMs')}"
            )

            # Create a button to download the data
            if st.button(
                "📥 Télécharger et préparer les données AOMs",
                key="btn_download",
            ):
                # Create a placeholder for progress updates
                status_placeholder = st.empty()
                progressbar_placeholder = st.empty()

                # Download and process data
                if download_and_process_aom_data(
                    dataset_aoms, status_placeholder, progressbar_placeholder
                ):
                    # Success message
                    st.success("✅ Données téléchargées avec succès!")

                    # Show the number of validated AOMs
                    if st.session_state.aoms_data:
                        st.info(
                            f"{len(st.session_state.aoms_data)} AOMs validées et prêtes à être mises à jour dans Grist"
                        )
        else:
            st.error("Impossible de récupérer les informations sur le dataset")

with st.expander(
    "💾 Task 3 : upload des données AOMs dans Grist", expanded=False
):
    # Add a button to update the data in Grist if we have validated AOMs
    if (
        "aoms_data" in st.session_state
        and len(st.session_state.aoms_data) > 0
        and not st.session_state.update_performed
    ):
        st.markdown(
            """
        Les données ont été validées et sont prêtes à être mises à jour dans Grist.
        Cliquez sur le bouton ci-dessous pour lancer la mise à jour.
        """
        )

        if st.button(
            "📤 Mettre à jour les données dans Grist", key="btn_update_grist"
        ):
            status_placeholder = st.empty()
            progressbar_placeholder = st.empty()

            # Update the data in Grist
            if asyncio.run(
                update_aoms_in_grist(
                    st.session_state.aoms_data,
                    status_placeholder,
                    progressbar_placeholder,
                )
            ):
                st.success("✅ Données mises à jour avec succès dans Grist!")
            else:
                st.error(
                    "❌ Erreur lors de la mise à jour des données dans Grist"
                )

    # Reset update flag
    if st.session_state.update_performed:
        st.session_state.update_performed = False

with st.expander(
    "↕️ Task 4 : mise à jour des données Passim / Comarquage", expanded=False
):
    st.markdown(
        """Les données sur les offres de transport proviennent de l'annuaire
    Passim du Cerema.
    """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
        **Guide étape par étape :**
        1. Cliquez sur le lien ci-dessous pour ouvrir le site dans un nouvel onglet
        2. Accédez à la section **Offre de transport**
        3. Cliquez sur le bouton "Exporter"
        4. Sélectionnez le format CSV
        5. Importez le fichier téléchargé dans la table **comarquage-offretransport** du grist
        """
        )

        st.markdown(
            f"""
            <div style="text-align: center; margin: 20px 0;">
                <a href="{URL_PASSIM_OFFRES_TRANSPORT}" target="_blank"
                    style="background-color: #4CAF50; color: white;
                            padding: 10px 20px; text-decoration: none;
                            border-radius: 5px; font-weight: bold;">
                    Ouvrir Passim dans un nouvel onglet
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        **Guide étape par étape :**
        1. Cliquez sur le lien ci-dessous pour ouvrir le site dans un nouvel onglet
        2. Accédez à la section **AOM**
        3. Cliquez sur le bouton "Exporter"
        4. Sélectionnez le format CSV
        5. Importez le fichier téléchargé dans la table **comarquage-aom** du grist
        """
        )

        st.markdown(
            f"""
            <div style="text-align: center; margin: 20px 0;">
                <a href="{URL_PASSIM_AOMS}" target="_blank"
                    style="background-color: #4CAF50; color: white;
                            padding: 10px 20px; text-decoration: none;
                            border-radius: 5px; font-weight: bold;">
                    Ouvrir Passim dans un nouvel onglet
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander(
    "🧹 Task 5 : filtre des offres de transport pertinentes", expanded=False
):
    st.markdown(
        """
        Les données sur les offres de transport concernent aussi bien le transport collectif,
        que le covoiturage, vélos, trotinettes etc. Par ailleurs, ces offres concercent des territoirers variés.
        Pour associer une AOM à une ou plusieurs offres de transport, il convient de :
            - filtrer les offres de transport pour ne garder que celles qui concernent le transport collectif
            - filtrer les offres de transport pour ne garder que celles qui concernent le territoire de l'AOM
        """
    )
    # Afficher les informations sur les dataframes
    col1, col2 = st.columns(2)
    with col1:
        offers_df = offers_df[
            (offers_df["type_de_transport"] == "Transport collectif régional")
            | (offers_df["type_de_transport"] == "Transport collectif urbain")
        ]
        offers_df = offers_df[offers_df["territoire_s_concerne_s"] != ""]
        offers_df = offers_df[offers_df["site_web_principal"] != ""]
        st.info(
            f"Nombre d'offres de transport collectif avec url renseignée: {len(offers_df)}"
        )
        st.info(
            f"Nombre de territoires uniques dans les offres de transport collectif: {offers_df['territoire_s_concerne_s'].nunique()}"
        )
    with col2:
        comarquage_aoms_df = comarquage_aoms_df[
            comarquage_aoms_df["territoire_s_concerne_s"] != ""
        ]
        st.info(
            f"Nombre d'AOMs du jeu de données **comarquage** avec territoires: {len(comarquage_aoms_df)}"
        )
        # Compter les territoires non-nuls
        non_null_territoires = comarquage_aoms_df[
            "territoire_s_concerne_s"
        ].dropna()
        st.info(
            f"Nombre de territoires uniques dans les AOMs: {non_null_territoires.nunique()}"
        )

    # 1. Jointure complète (FULL OUTER JOIN)
    st.subheader("1. Jointure complète (FULL OUTER JOIN)")
    st.markdown(
        "Cette jointure montre toutes les lignes des deux dataframes, avec des valeurs nulles lorsqu'il n'y a pas de correspondance."
    )

    # Effectuer la jointure
    full_join = pd.merge(
        offers_df,
        comarquage_aoms_df,
        on="territoire_s_concerne_s",
        how="outer",
        suffixes=("_offre", "_aom"),
    )

    # Afficher les résultats
    st.metric("Nombre total de lignes après jointure complète", len(full_join))
    # st.dataframe(full_join)

    # # 2. Offres sans correspondance dans les AOMs (LEFT JOIN + filtre)
    st.subheader("2. Offres sans correspondance dans les AOMs")
    st.markdown(
        "Cette jointure montre les offres de transport qui n'ont pas de correspondance dans les AOMs Comarquage."
    )

    # # Effectuer la jointure à gauche
    left_join = pd.merge(
        offers_df,
        comarquage_aoms_df,
        on="territoire_s_concerne_s",
        how="left",
        suffixes=("_offre", "_aom"),
    )

    # Filtrer pour ne garder que les lignes sans correspondance
    left_only = left_join[left_join["id2"].isna()]

    # # Afficher les résultats
    st.metric("Nombre d'offres sans correspondance", len(left_only))
    # st.dataframe(left_only)

    # # 3. AOMs sans correspondance dans les offres (RIGHT JOIN + filtre)
    st.subheader("3. AOMs sans correspondance dans les offres")
    st.markdown(
        "Cette jointure montre les AOMs Comarquage qui n'ont pas de correspondance dans les offres de transport."
    )

    # Effectuer la jointure à droite
    right_join = pd.merge(
        offers_df,
        comarquage_aoms_df,
        on="territoire_s_concerne_s",
        how="right",
        suffixes=("_offre", "_aom"),
    )

    # Filtrer pour ne garder que les lignes sans correspondance
    right_only = right_join[right_join["last_update_offre"].isna()]

    # Afficher les résultats
    st.metric("Nombre d'AOMs sans correspondance", len(right_only))
    # st.dataframe(right_only)

    # # 4. Jointure interne (INNER JOIN)
    st.subheader("4. Jointure interne (INNER JOIN)")
    st.markdown(
        "Cette jointure montre uniquement les lignes qui ont une correspondance dans les deux dataframes."
    )

    # Effectuer la jointure interne
    inner_join = pd.merge(
        offers_df,
        comarquage_aoms_df,
        on="territoire_s_concerne_s",
        how="inner",
        suffixes=("_offre", "_aom"),
    )

    inner_join["ndeg_siren"] = inner_join["ndeg_siren"].astype("Int64")
    aoms_df["n_siren_groupement"] = aoms_df["n_siren_groupement"].astype(
        "Int64"
    )

    # Effectuer la jointure
    aoms_with_offers = pd.merge(
        inner_join,
        aoms_df,
        left_on="ndeg_siren",
        right_on="n_siren_groupement",
        how="inner",
        suffixes=("_comarquage", "_aom"),
    )

    # Sélectionner les colonnes requises
    aoms_with_offers_selected = aoms_with_offers[
        [
            "n_siren_groupement",
            "n_siren_aom",
            "nom_aom",
            "commune_principale_aom",
            "nombre_commune_aom",
            "population_aom",
            "surface_km_2",
            "id_reseau_aom",
            "nom_commercial",
            "exploitant",
            "site_web_principal",
            "territoire_s_concerne_s",
            "type_de_contrat",
        ]
    ]
    # Afficher les résultats
    st.metric(
        "Nombre de correspondances trouvées", len(aoms_with_offers_selected)
    )
    st.dataframe(aoms_with_offers_selected)

aoms_avec_offres = aoms_with_offers_selected["n_siren_groupement"].nunique()
# Get unique AOMs first, then sum their population and surface to avoid counting multiple times
unique_aoms_with_offers = aoms_with_offers_selected.drop_duplicates(
    subset=["n_siren_groupement"]
)
population_avec_offres = unique_aoms_with_offers["population_aom"].sum()
surface_avec_offres = unique_aoms_with_offers["surface_km_2"].sum()
total_aoms = comarquage_aoms_df["ndeg_siren"].nunique()
total_population = comarquage_aoms_df["population_de_l_aom"].sum()
total_surface = comarquage_aoms_df["surface_km2"].sum()
# Calculate coverage rates
taux_couverture = (aoms_avec_offres / total_aoms) * 100
taux_population = (population_avec_offres / total_population) * 100
taux_surface = (surface_avec_offres / total_surface) * 100


with st.expander(
    "💾 Task 6 : upload AOMs avec offres de transport et leurs urls renseignées",
    expanded=False,
):
    # Row 1: Total AOMs and AOMs with offers
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Nombre total d'AOMs dans la base", value=f"{total_aoms}"
        )
    with col2:
        st.metric(
            label="AOMs ayant au moins une offre de transport",
            value=f"{aoms_avec_offres}",
            delta=f"{aoms_avec_offres}/{total_aoms}",
        )

    # Row 2: Coverage rates
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Taux de couverture des AOMs",
            value=f"{taux_couverture:.2f}%",
        )
    with col2:
        st.metric(
            label="Taux de couverture de la surface",
            value=f"{taux_surface:.2f}%",
        )
    with col3:
        st.metric(
            label="Taux de couverture de la population",
            value=f"{taux_population:.2f}%",
        )

    # Bouton pour télécharger les données jointes dans Grist
    if st.button(
        "📤 Mettre à jour les AOMs avec offres de transport dans Grist",
        key="btn_update_aoms_with_offers",
    ):
        status_placeholder = st.empty()
        progressbar_placeholder = st.empty()

        # Upload the data to Grist
        if asyncio.run(
            upload_aoms_with_offers(
                aoms_with_offers_selected,
                status_placeholder,
                progressbar_placeholder,
            )
        ):
            st.success("✅ Données mises à jour avec succès dans Grist!")
        else:
            st.error("❌ Erreur lors de la mise à jour des données dans Grist")
