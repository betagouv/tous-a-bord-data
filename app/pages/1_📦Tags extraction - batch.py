import asyncio
import os

import pandas as pd
import streamlit as st
from constants.keywords import DEFAULT_KEYWORDS
from models.grist_models import AomWithTags
from services.batch_tag_extraction import BatchProcessor
from services.grist_service import GristDataService
from services.llm_services import LLM_MODELS

# Configuration de la page
st.set_page_config(page_title="Traitement Batch", layout="wide", page_icon="🔄")

# Section de traitement batch
st.header("🔄 Traitement batch")
st.markdown(
    "Cette section permet de lancer un traitement batch pour plusieurs AOMs en utilisant les configurations définies ci-dessus."
)

# Initialiser les variables de session pour le batch
if "batch_processing_active" not in st.session_state:
    st.session_state.batch_processing_active = False
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

# Récupérer les configurations des expanders
batch_config = {
    "keywords": st.session_state.get(
        "selected_keywords", DEFAULT_KEYWORDS.copy()
    ),
    "model_name": st.session_state.get(
        "selected_model_name", list(LLM_MODELS.keys())[0]
    ),
}

# Afficher les configurations actuelles
st.subheader("Configuration actuelle")
col1, col2 = st.columns(2)
with col1:
    st.write("**Keywords pour scraping:**")
    st.write(", ".join(batch_config["keywords"]))
with col2:
    st.write("**Modèle pour classification:**")
    st.write(batch_config["model_name"])


async def get_aom_transport_offers():
    try:
        # Get GristDataService instance
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY")
        )
        doc_id = os.getenv("GRIST_DOC_INTERMEDIARY_ID")

        aoms = await grist_service.get_aom_transport_offers(doc_id)
        return aoms
    except Exception as e:
        st.error(f"Erreur lors de la connexion à Grist : {str(e)}")
        return []


aoms = asyncio.run(get_aom_transport_offers())

# Bouton pour lancer le traitement batch
if st.button(
    "🚀 Lancer le traitement batch", type="primary", use_container_width=True
):
    # Réinitialiser les résultats précédents
    st.session_state.batch_results = []
    st.session_state.batch_processing_active = True

    # Conteneurs pour l'affichage en temps réel
    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Fonction pour mettre à jour la progression
        def update_progress(current, total, result):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(
                f"Progression: {current}/{total} AOMs traités ({progress:.1%})"
            )

        batch_processor = BatchProcessor(max_workers=4)

        # Lancer le traitement batch
        with st.spinner("Traitement batch en cours..."):
            try:
                # Configurer le batch processor avec les paramètres des expanders
                batch_processor.keywords = batch_config["keywords"]
                batch_processor.model_name = batch_config["model_name"]

                # Lancer le traitement
                results = batch_processor.process_batch(
                    aom_list=aoms[6:10], progress_callback=update_progress
                )

                # Sauvegarder les résultats dans la session
                st.session_state.batch_results = results

                # Convertir les résultats en AomWithTags pour Grist
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

                        # Créer un objet AomWithTags
                        aom_with_tags = AomWithTags(
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
                            labels=result.tags,
                            fournisseurs=result.providers,
                            status=result.status,
                        )
                        aoms_with_tags.append(aom_with_tags)

                # Sauvegarder les résultats dans Grist
                if aoms_with_tags:

                    async def save_to_grist():
                        try:
                            # Get GristDataService instance
                            grist_service = GristDataService.get_instance(
                                api_key=os.getenv("GRIST_API_KEY")
                            )
                            doc_id = os.getenv("GRIST_DOC_OUTPUT_ID")

                            # Mettre à jour les AOMs avec tags dans Grist
                            await grist_service.update_aom_with_tags_batch(
                                aoms_with_tags, doc_id
                            )
                            return True
                        except Exception as e:
                            st.error(
                                f"Erreur lors de la sauvegarde dans Grist: {str(e)}"
                            )
                            return False

                    # Exécuter la fonction asynchrone
                    success = asyncio.run(save_to_grist())
                    if success:
                        st.success("✅ Résultats sauvegardés dans Grist")

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
            result.status, "❔"
        )

        results_data.append(
            {
                "SIREN": result.n_siren_aom,
                "Nom AOM": result.nom_aom,
                "Statut": f"{status_emoji} {result.status}",
                "Labels": ", ".join(result.tags) if result.tags else "",
                "Fournisseurs": ", ".join(result.providers)
                if result.providers
                else "",
                "Nb Labels": len(result.tags) if result.tags else 0,
                "Nb Fournisseurs": len(result.providers)
                if result.providers
                else 0,
                "Temps (s)": f"{result.processing_time:.1f}"
                if result.processing_time
                else "",
                "Erreur": result.error_message or "",
            }
        )

    results_df = pd.DataFrame(results_data)

    # Filtres pour le tableau
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filtrer par status",
            ["Tous"]
            + [
                s.replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", "")
                for s in results_df["Statut"].unique()
            ],
            key="batch_status_filter",
        )

    with col2:
        min_tags = st.number_input(
            "Nombre minimum de labels",
            min_value=0,
            value=0,
            key="batch_min_tags_filter",
        )

    with col3:
        search_term = st.text_input(
            "Rechercher dans le nom", key="batch_search_filter"
        )

    # Appliquer les filtres
    filtered_df = results_df.copy()

    if status_filter != "Tous":
        filtered_df = filtered_df[
            filtered_df["Statut"].str.contains(status_filter)
        ]

    if min_tags > 0:
        filtered_df = filtered_df[filtered_df["Nb Labels"] >= min_tags]

    if search_term:
        filtered_df = filtered_df[
            filtered_df["Nom AOM"].str.contains(
                search_term, case=False, na=False
            )
        ]

    # Afficher le tableau filtré
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Bouton pour relancer les AOMs en erreur
    if st.button("🔄 Relancer les AOMs en erreur", key="retry_batch_errors"):
        error_aoms = [
            result.n_siren_aom
            for result in st.session_state.batch_results
            if result.status == "error"
        ]

        if error_aoms:
            st.session_state.batch_processing_active = True

            # Initialiser le BatchProcessor
            from services.batch_tag_extraction import BatchProcessor

            batch_processor = BatchProcessor(max_workers=4)

            # Configurer le batch processor avec les paramètres des expanders
            batch_processor.keywords = batch_config["keywords"]
            batch_processor.model_name = batch_config["model_name"]

            # Conteneur pour la progression
            progress_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Fonction pour mettre à jour la progression
                def update_progress(current, total, result):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Progression: {current}/{total} AOMs traités ({progress:.1%})"
                    )

                # Lancer le traitement pour les AOMs en erreur
                with st.spinner(
                    f"Relancement de {len(error_aoms)} AOMs en erreur..."
                ):
                    try:
                        results = batch_processor.process_batch(
                            aom_list=error_aoms,
                            progress_callback=update_progress,
                        )

                        # Mettre à jour les résultats dans la session
                        # Remplacer les anciens résultats en erreur par les nouveaux
                        updated_results = []
                        for result in st.session_state.batch_results:
                            # Si c'était une AOM en erreur, chercher le nouveau résultat
                            if result.status == "error":
                                new_result = next(
                                    (
                                        r
                                        for r in results
                                        if r.n_siren_aom == result.n_siren_aom
                                    ),
                                    result,  # Garder l'ancien si pas trouvé
                                )
                                updated_results.append(new_result)
                            else:
                                # Garder les résultats qui n'étaient pas en erreur
                                updated_results.append(result)

                        st.session_state.batch_results = updated_results
                        st.success(
                            f"✅ Relancement terminé pour {len(error_aoms)} AOMs"
                        )
                        st.rerun()  # Recharger la page pour afficher les nouveaux résultats

                    except Exception as e:
                        st.error(f"❌ Erreur lors du relancement: {str(e)}")
                    finally:
                        st.session_state.batch_processing_active = False
        else:
            st.info("Aucune AOM en erreur à relancer")
