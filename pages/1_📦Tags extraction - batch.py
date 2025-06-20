import asyncio
import logging
import os

import pandas as pd
import streamlit as st

from constants.keywords import DEFAULT_KEYWORDS
from models.grist_models import AomTags
from services.batch_tag_extraction import BatchProcessor
from services.grist_service import GristDataService
from services.llm_services import LLM_MODELS


# utils
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


def update_progress(current, total, result=None):
    progress = current / total
    progress_bar.progress(progress)
    status_text.text(
        f"Progression: {current}/{total} AOMs traités ({progress:.1%})"
    )


async def save_to_grist(aoms_with_tags, status_placeholder, progress_bar):
    grist_service = GristDataService.get_instance(
        api_key=os.getenv("GRIST_API_KEY")
    )
    doc_id = os.getenv("GRIST_DOC_OUTPUT_ID")
    delete_result = await grist_service.delete_aom_tags(aoms_with_tags, doc_id)
    if delete_result.get("deleted", 0) > 0:
        st.info(f"{delete_result.get('deleted')} enregistrements supprimés")
    batch_size = 50

    batches = [
        aoms_with_tags[i : i + batch_size]
        for i in range(0, len(aoms_with_tags), batch_size)
    ]
    total_batches = len(batches)

    total_updated = 0
    for i, batch in enumerate(batches):
        st.info(
            f"Traitement du lot {i+1}/{total_batches} ({len(batch)} aoms par batch)"
        )

        await grist_service.update_aom_tags(batch, doc_id=doc_id)
        total_updated += len(batch)
        progress = min((i + 1) / total_batches, 1.0)
        progress_bar.progress(progress)

    progress_bar.progress(1.0)

    status_placeholder.success(
        f"Total des enregistrements mis à jour: {total_updated}"
    )
    progress_bar.empty()


# UI
if "page_loaded" not in st.session_state:
    st.session_state.page_loaded = True
    on_page_load()

st.set_page_config(
    page_title="Tags extraction - batch", layout="wide", page_icon="📦"
)

st.header("📦 Tags extraction - batch")
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

        batch_processor = BatchProcessor(st.session_state.loop, max_workers=4)

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

                # Lancer le traitement
                results = batch_processor.process_batch(
                    aom_list=aoms, progress_callback=update_progress
                )
                st.session_state.batch_results = results

                # Convertir les résultats en AomTags pour Grist
                aoms_with_tags = []
                for result in results:
                    if result.status == "success":
                        # Trouver l'AOM correspondante dans la liste complète pour récupérer les informations manquantes
                        aom_info = next(
                            (
                                a
                                for a in aoms
                                if str(a.n_siren_aom) == result.n_siren_aom
                            ),
                            None,
                        )

                        aom_with_tags = AomTags(
                            n_siren_groupement=int(result.n_siren_aom),
                            n_siren_aom=int(result.n_siren_aom),
                            nom_aom=result.nom_aom,
                            commune_principale_aom=aom_info.commune_principale_aom
                            if aom_info
                            else "",
                            nombre_commune_aom=aom_info.nombre_commune_aom
                            if aom_info
                            else 0,
                            population_aom=aom_info.population_aom
                            if aom_info
                            else None,
                            surface_km_2=aom_info.surface_km_2
                            if aom_info
                            else None,
                            id_reseau_aom=aom_info.id_reseau_aom
                            if aom_info
                            else None,
                            nom_commercial=aom_info.nom_commercial
                            if aom_info
                            else None,
                            exploitant=aom_info.exploitant
                            if aom_info
                            else None,
                            site_web_principal=aom_info.site_web_principal
                            if aom_info
                            else None,
                            territoire_s_concerne_s=aom_info.territoire_s_concerne_s
                            if aom_info
                            else None,
                            type_de_contrat=aom_info.type_de_contrat
                            if aom_info
                            else None,
                            criteres_eligibilite=result.tags,
                            fournisseurs=result.providers,
                        )
                        aoms_with_tags.append(aom_with_tags)
                st.session_state["aoms_with_tags"] = aoms_with_tags
                st.success(
                    f"✅ Traitement batch terminé pour {len(results)} AOMs"
                )
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement batch: {str(e)}")
            finally:
                st.session_state.batch_processing_active = False

# Afficher les résultats du traitement batch
if st.session_state.batch_results:
    st.subheader("Résultats du traitement batch")

    results_data = []
    for result in st.session_state.batch_results:
        # Emoji pour le statut
        status_emoji = {"success": "✅", "error": "❌", "no_data": "⚠️"}.get(
            result.status, ""
        )
        setattr(result, "status", f"{status_emoji} {result.status}")
        results_data.append(result)

    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df)

if st.session_state.aoms_with_tags:
    st.subheader(
        "✨ Jeu de données - Critères d'éligibilité aux tarifs sociaux et solidaires des transports"
    )
    aoms_dict_list = []
    for aom in st.session_state["aoms_with_tags"]:
        aom_dict = aom.model_dump()
        aoms_dict_list.append(aom_dict)
    st.dataframe(pd.DataFrame(aoms_dict_list))
    # TODO - fix this
    # if st.button(
    #     "📤 Mettre à jour les critères d'éligibilité aux tarifs sociaux et solidaires des transports dans Grist",
    #     key="btn_update_aoms_with_tags",
    # ):
    #     # Sauvegarder les résultats dans Grist
    #     status_placeholder = st.empty()
    #     progressbar_placeholder = st.empty()
    #     success = asyncio.run(
    #         save_to_grist(
    #             st.session_state["aoms_with_tags"],
    #             status_placeholder,
    #             progressbar_placeholder,
    #         )
    #     )
    #     if success:
    #         status_placeholder.success("✅ Résultats sauvegardés dans Grist")
