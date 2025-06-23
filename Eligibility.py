import asyncio
import logging
import os

import nest_asyncio
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Initialize the event loop before importing crawl4ai
# flake8: noqa: E402
nest_asyncio.apply()

from constants.keywords import DEFAULT_KEYWORDS
from models.grist_models import AomTags
from services.batch_tag_extraction import BatchProcessor
from services.grist_service import GristDataService
from services.llm_services import LLM_MODELS
from utils.dataframe_utils import filter_dataframe

FLY_API_TOKEN = os.getenv("FLY_API_TOKEN")
APP_NAME = "tous-a-bord-data"

load_dotenv()


# utils
async def fetch_aoms_from_grist():
    """Charge les données des AOM depuis Grist."""
    try:
        # Utilisation du singleton pattern
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY"),
        )
        doc_id = os.getenv("GRIST_DOC_OUTPUT_ID")
        aoms = await grist_service.get_aom_tags(doc_id)

        # Filtrer les tags "L" des criteres_eligibilite et fournisseurs
        for aom in aoms:
            if aom.criteres_eligibilite:
                aom.criteres_eligibilite = [
                    tag for tag in aom.criteres_eligibilite if tag != "L"
                ]
            if aom.fournisseurs:
                aom.fournisseurs = [
                    fournisseur
                    for fournisseur in aom.fournisseurs
                    if fournisseur != "L"
                ]

        return pd.DataFrame([aom.model_dump() for aom in aoms])
    except Exception as e:
        st.error(f"Erreur lors du chargement des AOM depuis Grist: {str(e)}")
        return pd.DataFrame()


def on_page_load():
    if "batch_processing_active" not in st.session_state:
        st.session_state.batch_processing_active = False
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "aoms_with_tags" not in st.session_state:
        st.session_state.aoms_with_tags = []

    if "available_keywords" not in st.session_state:
        st.session_state.available_keywords = DEFAULT_KEYWORDS.copy()

    # Initialiser la configuration dans session_state
    if "batch_config" not in st.session_state:
        st.session_state.batch_config = {
            "keywords": st.session_state.get(
                "selected_keywords", DEFAULT_KEYWORDS.copy()
            ),
            "model_name": st.session_state.get(
                "selected_model_name", list(LLM_MODELS.keys())[0]
            ),
        }
    logging.info("Page 'Tags extraction - batch' chargée et initialisée")


def change_config():
    st.session_state.batch_config = {
        "keywords": st.session_state.selected_keywords,
        "model_name": st.session_state.selected_model_name,
    }


async def get_aom_transport_offers():
    try:
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY")
        )
        doc_id = os.getenv("GRIST_DOC_INTERMEDIARY_ID")

        aoms = await grist_service.get_aom_transport_offers(doc_id)
        return aoms
    except Exception as e:
        st.error(f"Erreur lors de la connexion à Grist : {str(e)}")
        return []


def update_progress(current, total, result):
    progress = current / total
    progress_bar.progress(progress)
    status_text.text(
        f"Progression: {current}/{total} AOMs traités ({progress:.1%})"
    )
    st.write(
        f"**Résultat pour l'AOM {result.nom_aom}** ({result.n_siren_aom})"
    )
    st.write(result)


# UI
st.set_page_config(
    page_title="Critères d'éligibilité à la tarification sociale et solidaire des transports",
    page_icon="🚌",
    layout="wide",
)
st.subheader(
    "Critères d'éligibilité à la tarification sociale et solidaire des transports"
)

try:
    if "aoms_data" not in st.session_state:
        st.session_state.aoms_data = asyncio.run(fetch_aoms_from_grist())
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {str(e)}")
    st.session_state.aoms_data = pd.DataFrame()

# Search bar
search_term = st.text_input(
    "🔍 Rechercher dans toutes les colonnes",
    placeholder="Exemple : Bordeaux Métropole",
)

# Filter the data
filtered_df = filter_dataframe(st.session_state.aoms_data, search_term)

# Display the number of results
nb_results = len(filtered_df)

# Safely calculate population
try:
    # Check if filtered_df is a DataFrame and has the population_aom column
    if (
        isinstance(filtered_df, pd.DataFrame)
        and "population_aom" in filtered_df.columns
    ):
        # Filter out None values and convert to numeric
        population_values = pd.to_numeric(
            filtered_df["population_aom"].dropna(), errors="coerce"
        )
        population = population_values.sum()
    else:
        # If filtered_df is not a DataFrame or doesn't have the column, set population to 0
        population = 0
except Exception as e:
    st.warning(f"Erreur lors du calcul de la population: {str(e)}")
    population = 0

first = f"**{nb_results} AOM{'s' if nb_results > 1 else ''}**"
second = f"trouvée{'s' if nb_results > 1 else ''}"
taux_population = population * 100 / 66647129 if population > 0 else 0
st.write(
    f"📊{first} {second}, soit " f"**{taux_population:.2f}% de la population**"
)

# Display the filtered table
st.dataframe(
    filtered_df,
    hide_index=True,
)


# UI
if "page_loaded" not in st.session_state:
    st.session_state.page_loaded = True
    on_page_load()

st.divider()
st.subheader("Extraction des critères d'éligibilité - Batch")

st.markdown(
    "Cette section permet de lancer un traitement batch pour plusieurs AOMs en utilisant les configurations définies ci-dessus."
)

if "batch_processing_active" not in st.session_state:
    st.session_state.batch_processing_active = False
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []


st.subheader("Configuration")

if "available_keywords" not in st.session_state:
    st.session_state.available_keywords = DEFAULT_KEYWORDS.copy()


new_keyword = st.text_input(
    "Ajouter un nouveau mot-clé :",
    placeholder="Entrez un nouveau mot-clé et appuyez sur Entrée",
    help="Le nouveau mot-clé sera ajouté à la liste disponible",
)

if new_keyword:
    if new_keyword not in st.session_state.available_keywords:
        st.session_state.available_keywords.append(new_keyword)
        # Mettre à jour batch_config directement au lieu de selected_keywords
        if (
            "batch_config" in st.session_state
            and "keywords" in st.session_state.batch_config
        ):
            if new_keyword not in st.session_state.batch_config["keywords"]:
                st.session_state.batch_config["keywords"].append(new_keyword)
        st.rerun()

selected_keywords = st.multiselect(
    "Mots-clés :",
    options=st.session_state.available_keywords,
    default=st.session_state.batch_config.get(
        "keywords", DEFAULT_KEYWORDS.copy()
    ),
    key="selected_keywords",
    on_change=change_config,
)

st.write("**Modèle pour classification:**")
selected_model_name = st.selectbox(
    "Modèle LLM à utiliser",
    options=list(LLM_MODELS.keys()),
    index=0,
    key="selected_model_name",
    on_change=change_config,
)

aoms = asyncio.run(get_aom_transport_offers())

# Sélection des AOMs à traiter
st.subheader("Sélection des AOMs à traiter")

# Créer un DataFrame pour afficher les AOMs de manière plus lisible
if aoms:
    aom_df = pd.DataFrame(
        [
            {
                "SIREN": aom.n_siren_aom,
                "Nom AOM": aom.nom_aom,
                "Commune principale": getattr(
                    aom, "commune_principale_aom", ""
                ),
                "Site web": getattr(aom, "site_web_principal", ""),
            }
            for aom in aoms
        ]
    )

    # Créer une liste d'options pour le multiselect des AOMs
    aom_options = [f"{aom.nom_aom} ({aom.n_siren_aom})" for aom in aoms]

    # Obtenir les indices des lignes filtrées dans le DataFrame original
    # Utiliser le filtered_df déjà défini plus haut dans le code
    filtered_indices = list(range(len(aom_options)))
    if not filtered_df.empty:
        # Récupérer les SIREN des AOMs filtrées
        filtered_sirens = (
            filtered_df["n_siren_groupement"].astype(str).tolist()
            if "n_siren_groupement" in filtered_df.columns
            else []
        )
        # Trouver les indices des AOMs correspondantes
        filtered_indices = [
            i
            for i, aom in enumerate(aoms)
            if str(aom.n_siren_aom) in filtered_sirens
        ]

    # Boutons pour sélectionner/désélectionner toutes les AOMs filtrées
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Sélectionner toutes les AOMs filtrées", use_container_width=True
        ):
            if "selected_aoms" not in st.session_state:
                st.session_state.selected_aoms = []
            st.session_state.selected_aoms = list(
                set(st.session_state.selected_aoms + filtered_indices)
            )
            st.rerun()

    with col2:
        if st.button(
            "Désélectionner toutes les AOMs", use_container_width=True
        ):
            if "selected_aoms" in st.session_state:
                st.session_state.selected_aoms = []
                st.rerun()

    # Multiselect pour choisir les AOMs
    selected_aom_indices = st.multiselect(
        "Sélectionnez les AOMs à traiter :",
        options=filtered_indices,
        format_func=lambda i: aom_options[i],
        key="selected_aoms",
        help="Vous pouvez sélectionner plusieurs AOMs en maintenant la touche Ctrl (ou Cmd sur Mac) enfoncée.",
    )

# Afficher le nombre d'AOMs sélectionnées
if selected_aom_indices:
    st.info(f"📊 {len(selected_aom_indices)} sites d'AOM(s) sélectionné(s)")

# init crawler event loop
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()

if st.button(
    "🚀 Lancer le traitement batch", type="primary", use_container_width=True
):
    st.session_state.batch_results = []
    st.session_state.batch_processing_active = True

    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        # init crawler event loop
        loop = st.session_state.loop
        batch_processor = BatchProcessor(max_workers=1)

        # Lancer le traitement batch
        with st.spinner("Traitement batch en cours..."):
            try:
                # Configurer le batch processor avec les paramètres de la session_state
                batch_processor.keywords = st.session_state.batch_config[
                    "keywords"
                ]
                batch_processor.model_name = st.session_state.batch_config[
                    "model_name"
                ]

                # Filtrer les AOMs sélectionnées
                selected_aoms = [aoms[i] for i in selected_aom_indices]

                if not selected_aoms:
                    st.warning(
                        "⚠️ Aucune AOM sélectionnée. Veuillez sélectionner au moins une AOM."
                    )
                    st.session_state.batch_processing_active = False

                # Lancer le traitement en utilisant le loop existant
                results = loop.run_until_complete(
                    batch_processor.process_batch(
                        aom_list=selected_aoms,
                        progress_callback=update_progress,
                    )
                )
                st.session_state.batch_results = results
                st.success(
                    f"✅ Traitement batch terminé pour {len(results)} AOMs"
                )
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement batch: {str(e)}")
            finally:
                st.session_state.batch_processing_active = False
