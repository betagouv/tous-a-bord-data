import asyncio
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import pandas as pd
import pyexcel_ods3
import requests
import streamlit as st
from constants.urls import URL_PASSIM
from dotenv import load_dotenv
from models.grist_models import Aom, Commune
from pydantic import BaseModel
from services.grist_service import GristDataService
from services.transport_gouv_client import get_aom_dataset
from utils.parser_utils import format_column

# Page configuration
st.set_page_config(
    page_title="Mise à jour de la BDD",
    page_icon="🔄",
    layout="wide",
)

# Load environment variables
load_dotenv()

# Initialize session state
if "update_performed" not in st.session_state:
    st.session_state.update_performed = False


def download_file(url: str, progress_placeholder) -> Optional[str]:
    """
    Download a file with progress tracking

    Args:
        url: URL to download
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        Path to downloaded file or None if error
    """
    progress_placeholder.info("Téléchargement du fichier en cours...")

    # Use a session for better control
    session = requests.Session()

    # First make a HEAD request to get the content length
    head_response = session.head(url)
    total_size = int(head_response.headers.get("content-length", 0))

    # Now make the actual GET request with streaming
    response = session.get(url, stream=True)
    if response.status_code != 200:
        progress_placeholder.error(
            f"Erreur de téléchargement: {response.status_code}"
        )
        return None

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_f:
        # Download with progress tracking
        downloaded = 0
        progress_bar = progress_placeholder.progress(0)

        for chunk in response.iter_content(chunk_size=4096):
            if chunk:  # filter out keep-alive chunks
                tmp_f.write(chunk)
                downloaded += len(chunk)

                # Update progress bar
                if total_size > 0:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress)

                    # Also show download speed and percentage
                    progress_text = f"Téléchargement: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB ({progress*100:.1f}%)"
                    progress_placeholder.info(progress_text)

        # Download complete
        progress_bar.progress(1.0)
        progress_placeholder.success("Téléchargement terminé!")
        return tmp_f.name


def process_ods_file(
    file_path: str, progress_placeholder
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process ODS file directly using pyexcel_ods3

    Args:
        file_path: Path to the ODS file
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        Tuple of (AOM data list, communes data list)
    """
    # Create a processing progress tracker
    processing_status = progress_placeholder.empty()
    processing_status.info("Traitement direct du fichier ODS...")
    processing_progress = progress_placeholder.progress(0)

    try:
        # Step 1: Load the ODS file directly with pyexcel_ods3 (much faster than pandas)
        processing_status.info("Chargement du fichier ODS...")
        processing_progress.progress(0.1)

        # Load the ODS file directly
        data = pyexcel_ods3.get_data(file_path)

        # Check if we have at least 2 sheets
        if len(data) < 2:
            processing_status.error(
                "Le fichier doit contenir deux feuilles (AOM et communes)"
            )
            return [], []

        # Get sheet names (keys of the data dictionary)
        sheet_names = list(data.keys())

        # Step 2: Process AOM data directly
        processing_status.info("Traitement des données AOM...")
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

        # Step 3: Process Communes data directly
        processing_status.info("Traitement des données Communes...")
        processing_progress.progress(0.6)

        # Get Communes data
        communes_sheet = data[sheet_names[1]]

        # Extract headers and format them
        communes_headers = [
            format_column(header) for header in communes_sheet[0]
        ]

        # Process Communes records directly
        communes_data = []
        for row in communes_sheet[1:]:  # Skip header row
            # Create a dictionary from headers and row values
            row_dict = {}
            for j, header in enumerate(communes_headers):
                if j < len(row):
                    row_dict[header] = row[j]
                else:
                    row_dict[header] = None

            # Store the raw data
            communes_data.append(row_dict)

        # Complete
        processing_progress.progress(1.0)
        processing_status.success("Traitement terminé avec succès!")

        return aom_data, communes_data
    except Exception as e:
        processing_status.error(
            f"Erreur lors du traitement du fichier: {str(e)}"
        )
        return [], []


T = TypeVar("T", bound=BaseModel)


def validate_data(
    raw_data: List[Dict], progress_placeholder, model_class: Type[T]
) -> List[T]:
    """
    Validate raw AOM data with Pydantic - simplified version

    Args:
        raw_data: List of dictionaries containing AOM data
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        List of validated Aom objects
    """
    progress_placeholder.info(
        f"Validation de {len(raw_data)} enregistrements avec Pydantic..."
    )
    validated_data = []
    errors = 0

    # Simple progress bar
    progress_bar = progress_placeholder.progress(0)

    # Process in batches for better performance
    batch_size = 100
    for i in range(0, len(raw_data), batch_size):
        batch = raw_data[i : i + batch_size]

        # Process each item in batch
        for row_dict in batch:
            try:
                # Direct validation with Pydantic
                data = model_class.model_validate(row_dict)
                validated_data.append(data)
            except Exception:
                errors += 1

        # Update progress after each batch
        progress = min((i + batch_size) / len(raw_data), 1.0)
        progress_bar.progress(progress)

    # Complete progress
    progress_bar.progress(1.0)
    progress_placeholder.success(
        f"Validation terminée: {len(validated_data)} objets valides, {errors} erreurs"
    )

    return validated_data


async def update_aoms_in_grist(aoms: List[Aom], progress_placeholder) -> bool:
    """
    Update AOM data in Grist

    Args:
        aoms: List of Aom objects
        progress_placeholder: Streamlit placeholder for progress updates

    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Get GristDataService instance
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
            doc_id=os.getenv("GRIST_DOC_INPUTDATA_ID"),
        )

        # Create progress tracking
        status_text = progress_placeholder.empty()
        status_text.info("Mise à jour des données dans Grist...")
        progress_bar = progress_placeholder.progress(0)

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
            status_text.info(
                f"Traitement du lot {i+1}/{total_batches} ({len(batch)} enregistrements)"
            )

            # Update AOM data for this batch
            result = await grist_service.update_aoms(batch)

            # Check if update was successful
            if result and isinstance(result, dict):
                total_updated += len(batch)

                # Update progress
                progress = min((i + 1) / total_batches, 1.0)
                progress_bar.progress(progress)
            else:
                progress_placeholder.error(
                    f"Échec du traitement du lot {i+1}/{total_batches}"
                )
                return False

        # Complete progress
        progress_bar.progress(1.0)
        status_text.success(
            f"Total des enregistrements mis à jour: {total_updated}"
        )

        return True
    except Exception as e:
        progress_placeholder.error(
            f"Erreur lors de la mise à jour des données AOM: {str(e)}"
        )
        return False


async def fetch_current_data():
    """
    Fetch current data from Grist

    Returns:
        Tuple of (aoms, communes, transport_offers)
    """
    try:
        # Get GristDataService instance
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
            doc_id=os.getenv("GRIST_DOC_INPUTDATA_ID"),
        )

        # Fetch data
        aoms = await grist_service.get_aoms()
        communes = await grist_service.get_communes()
        transport_offers = await grist_service.get_transport_offers()

        return aoms, communes, transport_offers
    except Exception as e:
        st.error(
            f"Erreur lors de la récupération des données depuis Grist: {str(e)}"
        )
        return [], [], []


def download_and_process_aom_data(dataset_info, progress_placeholder):
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
        progress_placeholder.error(
            "URL non trouvée dans les informations du dataset"
        )
        return False

    # Download file
    file_path = download_file(url, progress_placeholder)
    if not file_path:
        return False

    try:
        # Process file
        aom_data, communes_data = process_ods_file(
            file_path, progress_placeholder
        )

        # Check if data was processed successfully
        if not aom_data or not communes_data:
            progress_placeholder.error("Erreur lors du traitement des données")
            return False

        # Validate AOM data with Pydantic immediately
        progress_placeholder.info("Validation des données avec Pydantic...")
        validated_aoms = validate_data(aom_data, progress_placeholder, Aom)
        validated_communes = validate_data(
            communes_data, progress_placeholder, Commune
        )

        # Store validated objects in session state
        st.session_state.aoms_data = validated_aoms
        st.session_state.communes_data = validated_communes

        return True
    except Exception as e:
        progress_placeholder.error(
            f"Erreur lors du traitement des données: {str(e)}"
        )
        return False
    finally:
        # Clean up temporary file
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)


# Main UI
st.header("🔄 Mise à jour des données d'entrée")
st.markdown(
    """
    - Cette page permet de mettre à jour les données des AOMs (Autorités Organisatrices de la Mobilité),
    les données de Communes et des offres de transport.
    - Les données sont récupérées depuis le site **transport.gouv.fr** et l'annuaire **Passim** du Cerema.
    """
)

# Current data section
with st.expander(
    "Données actuellement disponibles dans Grist", expanded=False
):
    # Fetch current data
    aoms, communes, transport_offers = asyncio.run(fetch_current_data())

    # Display counts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AOMs", len(aoms))
    with col2:
        st.metric("Communes", len(communes))
    with col3:
        st.metric("Offres de transport", len(transport_offers))

    # Display data tables
    if aoms:
        st.subheader("Table AOMs")
        aoms_df = pd.DataFrame([aom.model_dump() for aom in aoms])
        st.dataframe(aoms_df)

    if communes:
        st.subheader("Table Communes")
        communes_df = pd.DataFrame(
            [commune.model_dump() for commune in communes]
        )
        st.dataframe(communes_df)

    if transport_offers:
        st.subheader("Table Passim")
        offers_df = pd.DataFrame(
            [offer.model_dump() for offer in transport_offers]
        )
        st.dataframe(offers_df)

# AOM data section
st.subheader("Mise à jour des données AOMs et Communes")
st.markdown(
    """Les données des AOMs (Autorités Organisatrices de la Mobilité), et des
communes proviennent du **CEREMA** et sont formattées par **transport.gouv.fr**.
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
            f"Dataset disponible: {dataset_aoms.get('title', 'Données AOMs / Communes')}"
        )

        # Create a button to download the data
        if st.button(
            "📥 Télécharger et préparer les données AOMs / Communes",
            key="btn_download",
        ):
            # Create a placeholder for progress updates
            progress_placeholder = st.empty()

            # Download and process data
            if download_and_process_aom_data(
                dataset_aoms, progress_placeholder
            ):
                # Success message
                st.success("✅ Données téléchargées avec succès!")
    else:
        st.error("Impossible de récupérer les informations sur le dataset")


# Passim section
st.subheader("Mise à jour des données Passim")
st.markdown(
    """Les données sur les offres de transport proviennent de l'annuaire
Passim du Cerema.
"""
)

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

# Reset update flag
if st.session_state.update_performed:
    st.session_state.update_performed = False
