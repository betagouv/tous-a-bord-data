import threading
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from services.batch_tag_extraction import BatchProcessor

# Configuration de la page
st.set_page_config(page_title="Traitement Batch", layout="wide", page_icon="🔄")

st.title("🔄 Traitement Batch des AOMs")
st.markdown("---")

# Initialiser le batch processor
if "batch_processor" not in st.session_state:
    st.session_state.batch_processor = BatchProcessor()

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Nombre de workers
    max_workers = st.slider(
        "Nombre de workers parallèles",
        min_value=1,
        max_value=8,
        value=4,
        help="Plus de workers = traitement plus rapide, "
        "mais plus de charge système",
    )
    st.session_state.batch_processor.max_workers = max_workers

    # Options de filtrage
    st.subheader("🎯 Filtres")

    # Limite de traitement pour les tests
    limit_aoms = st.number_input(
        "Limiter le nombre d'AOMs (0 = toutes)",
        min_value=0,
        value=0,
        help="Utile pour tester sur un échantillon",
    )

# Vue d'ensemble des AOMs
st.header("📊 Vue d'ensemble des AOMs")

if st.button("🔍 Charger la liste des AOMs", key="load_aoms"):
    with st.spinner("Chargement des AOMs..."):
        aoms = st.session_state.batch_processor.get_all_aoms()
        st.session_state.aoms_list = aoms

if "aoms_list" in st.session_state:
    aoms_df = pd.DataFrame(
        st.session_state.aoms_list, columns=["SIREN", "Nom AOM", "Nb Sources"]
    )

    if limit_aoms > 0:
        aoms_df = aoms_df.head(limit_aoms)

    # Tableau des AOMs
    st.subheader("📋 Liste des AOMs à traiter")
    st.dataframe(aoms_df, use_container_width=True)

    st.session_state.filtered_aoms = [
        (row["SIREN"], row["Nom AOM"], row["Nb Sources"])
        for _, row in aoms_df.iterrows()
    ]

# Section de lancement du traitement
st.header("🚀 Lancement du traitement")

if "filtered_aoms" in st.session_state:
    nb_aoms_to_process = len(st.session_state.filtered_aoms)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(
            f"Prêt à traiter **{nb_aoms_to_process} AOMs** "
            f"avec **{max_workers} workers**"
        )

    with col2:
        if st.button(
            "🚀 Lancer le traitement batch",
            type="primary",
            use_container_width=True,
        ):
            # Réinitialiser les résultats précédents
            if "batch_results" in st.session_state:
                del st.session_state.batch_results

            # Arrêter tout traitement précédent
            st.session_state.processing_active = False

            # Lancement du traitement
            start_time = time.time()

            # Conteneurs pour l'affichage en temps réel
            progress_container = st.container()
            details_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                current_aom_text = st.empty()

            # Fonction de callback pour la progress bar principale
            def update_progress(current, total, result):
                progress = current / total
                progress_bar.progress(progress)
                status_text.text(
                    f"Progression: {current}/{total} AOMs traités "
                    f"({progress:.1%})"
                )
                current_aom_text.text(
                    f"Dernier traité: {result.nom_aom} - {result.status}"
                )

            # Fonction de callback pour les étapes détaillées
            def update_step(siren, nom_aom, step, progress):
                # Cette fonction est appelée pour chaque étape de chaque AOM
                # On peut l'utiliser pour des logs détaillés si nécessaire
                pass

            # Démarrer l'affichage des détails en temps réel
            st.session_state.processing_active = True
            details_placeholder = details_container.empty()

            def update_details_display():
                """Thread function pour mettre à jour l'affichage"""
                while st.session_state.get("processing_active", False):
                    try:
                        b_p = st.session_state.batch_processor
                        status_dict = b_p.get_processing_status()
                        if status_dict:
                            # Créer un DataFrame pour l'affichage
                            status_data = []
                            for status in status_dict.values():
                                # Déterminer l'emoji du statut
                                status_emoji = {
                                    "processing": "🔄",
                                    "success": "✅",
                                    "error": "❌",
                                    "no_data": "⚠️",
                                }.get(status.status, "🔄")

                                # Barre de progression visuelle
                                progress_bar_visual = "█" * int(
                                    status.progress * 10
                                ) + "░" * (10 - int(status.progress * 10))

                                status_data.append(
                                    {
                                        "Statut": f"{status_emoji} "
                                        f"{status.status}",
                                        "AOM": status.nom_aom[:25] + "..."
                                        if len(status.nom_aom) > 25
                                        else status.nom_aom,
                                        "SIREN": status.aom_id,
                                        "Étape courante": status.current_step[
                                            :40
                                        ]
                                        + "..."
                                        if len(status.current_step) > 40
                                        else status.current_step,
                                        "Progression": f"{progress_bar_visual}"
                                        f"{status.progress:.0%}",
                                    }
                                )

                            if status_data:
                                df_status = pd.DataFrame(status_data)

                                with details_placeholder.container():
                                    st.subheader(
                                        "🔄 Détails des traitements en cours"
                                    )
                                    st.dataframe(
                                        df_status,
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                                    # Statistiques rapides
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        processing_count = len(
                                            [
                                                s
                                                for s in status_dict.values()
                                                if s.status == "processing"
                                            ]
                                        )
                                        st.metric("En cours", processing_count)
                                    with col2:
                                        success_count = len(
                                            [
                                                s
                                                for s in status_dict.values()
                                                if s.status == "success"
                                            ]
                                        )
                                        st.metric("Terminés", success_count)
                                    with col3:
                                        error_count = len(
                                            [
                                                s
                                                for s in status_dict.values()
                                                if s.status == "error"
                                            ]
                                        )
                                        st.metric("Erreurs", error_count)
                                    with col4:
                                        avg_progress = (
                                            sum(
                                                s.progress
                                                for s in status_dict.values()
                                            )
                                            / len(status_dict.values())
                                            if status_dict
                                            else 0
                                        )
                                        st.metric(
                                            "Progression moy.",
                                            f"{avg_progress:.0%}",
                                        )

                        time.sleep(1)  # Mise à jour toutes les secondes
                    except Exception:
                        # En cas d'erreur dans l'affichage,
                        # on continue silencieusement
                        time.sleep(2)

            # Démarrer le thread d'affichage des détails
            details_thread = threading.Thread(
                target=update_details_display, daemon=True
            )
            details_thread.start()

            # Lancer le traitement principal
            with st.spinner("Traitement en cours..."):
                try:
                    results = st.session_state.batch_processor.process_batch(
                        aom_list=st.session_state.filtered_aoms,
                        progress_callback=update_progress,
                        step_callback=update_step,
                    )
                except Exception as e:
                    st.error(f"Erreur lors du traitement batch: {str(e)}")
                    results = []
                finally:
                    # Arrêter le thread de mise à jour
                    st.session_state.processing_active = False

            end_time = time.time()
            processing_duration = end_time - start_time

            # Nettoyer l'affichage des détails
            details_placeholder.empty()

            # Sauvegarder les résultats
            st.session_state.batch_results = results
            st.session_state.processing_duration = processing_duration

            # Affichage du succès
            st.success(
                f"✅ Traitement terminé en {processing_duration:.1f} "
                "secondes!"
            )

            # Sauvegarder en base de données
            try:
                st.session_state.batch_processor.save_results_to_db(results)
                st.success("💾 Résultats sauvegardés en base de données")
            except Exception as e:
                st.error(f"❌ Erreur lors de la sauvegarde: {str(e)}")

# Affichage des résultats
if "batch_results" in st.session_state:
    st.header("📈 Résultats du traitement")

    results = st.session_state.batch_results
    stats = st.session_state.batch_processor.get_summary_stats(results)

    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("AOMs traitées", stats["total_aoms"])
    with col2:
        st.metric(
            "Succès", stats["success_count"], f"{stats['success_rate']:.1f}%"
        )
    with col3:
        st.metric("Erreurs", stats["error_count"])
    with col4:
        st.metric("Sans données", stats["no_data_count"])
    with col5:
        st.metric("Temps moyen", f"{stats['avg_processing_time']:.1f}s")

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        # Graphique de statut
        status_counts = {
            "Succès": stats["success_count"],
            "Erreurs": stats["error_count"],
            "Sans données": stats["no_data_count"],
        }

        fig_status = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Répartition des statuts de traitement",
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col2:
        # Top 10 des tags
        if stats["most_common_tags"]:
            tags_df = pd.DataFrame(
                stats["most_common_tags"], columns=["Tag", "Occurrences"]
            )
            fig_tags = px.bar(
                tags_df,
                x="Occurrences",
                y="Tag",
                orientation="h",
                title="Top 10 des tags les plus fréquents",
            )
            fig_tags.update_layout(height=400)
            st.plotly_chart(fig_tags, use_container_width=True)

    # Tableau détaillé des résultats
    st.subheader("📋 Résultats détaillés")

    # Convertir les résultats en DataFrame
    results_data = []
    for result in results:
        # Emoji pour le statut
        status_emoji = {"success": "✅", "error": "❌", "no_data": "⚠️"}.get(
            result.status, "❔"
        )

        results_data.append(
            {
                "SIREN": result.n_siren_aom,
                "Nom AOM": result.nom_aom,
                "Statut": f"{status_emoji} {result.status}",
                "Tags": ", ".join(result.tags) if result.tags else "",
                "Fournisseurs": ", ".join(result.providers)
                if result.providers
                else "",
                "Nb Tags": len(result.tags) if result.tags else 0,
                "Nb Fournisseurs": len(result.providers)
                if result.providers
                else 0,
                "Temps (s)": f"{result.processing_time:.1f}"
                if result.processing_time
                else "",
                "Tokens Original": result.nb_tokens_original,
                "Tokens Filtrés": result.nb_tokens_filtered,
                "Nb Sources": result.nb_sources,
                "Dernière étape": result.current_step or "",
                "Erreur": result.error_message or "",
            }
        )

    results_df = pd.DataFrame(results_data)

    # Filtres pour le tableau
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filtrer par statut",
            ["Tous"]
            + [
                s.replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", "")
                for s in results_df["Statut"].unique()
            ],
            key="status_filter",
        )

    with col2:
        min_tags = st.number_input(
            "Nombre minimum de tags",
            min_value=0,
            value=0,
            key="min_tags_filter",
        )

    with col3:
        search_term = st.text_input(
            "Rechercher dans le nom", key="search_filter"
        )

    # Appliquer les filtres
    filtered_df = results_df.copy()

    if status_filter != "Tous":
        filtered_df = filtered_df[
            filtered_df["Statut"].str.contains(status_filter)
        ]

    if min_tags > 0:
        filtered_df = filtered_df[filtered_df["Nb Tags"] >= min_tags]

    if search_term:
        filtered_df = filtered_df[
            filtered_df["Nom AOM"].str.contains(
                search_term, case=False, na=False
            )
        ]

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Boutons d'export
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Télécharger CSV",
            csv,
            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key="download_csv",
        )

    with col2:
        # Bouton pour relancer des AOMs spécifiques
        if st.button("🔄 Relancer les AOMs en erreur", key="retry_errors"):
            error_aoms = [
                (result.n_siren_aom, result.nom_aom, result.nb_sources)
                for result in results
                if result.status == "error"
            ]

            if error_aoms:
                st.session_state.filtered_aoms = error_aoms
                st.info(f"Prêt à relancer {len(error_aoms)} AOMs en erreur")
                st.rerun()
            else:
                st.info("Aucune AOM en erreur à relancer")

    with col3:
        # Bouton pour relancer les AOMs sans données
        if st.button("⚠️ Relancer les AOMs sans données", key="retry_no_data"):
            no_data_aoms = [
                (result.n_siren_aom, result.nom_aom, result.nb_sources)
                for result in results
                if result.status == "no_data"
            ]

            if no_data_aoms:
                st.session_state.filtered_aoms = no_data_aoms
                st.info(
                    f"Prêt à relancer {len(no_data_aoms)} AOMs sans données"
                )
                st.rerun()
            else:
                st.info("Aucune AOM sans données à relancer")

else:
    st.info("👆 Chargez d'abord la liste des AOMs pour commencer")

# Section d'aide
with st.expander("❓ Aide et informations"):
    st.markdown(
        """
    ### Comment utiliser le traitement batch

    1. **Chargez la liste des AOMs** avec le bouton "Charger la liste des AOMs"
    2. **Configurez les paramètres** dans la barre latérale :
       - Nombre de workers pour la parallélisation
       - Limite pour les tests
    3. **Lancez le traitement** avec le bouton "Lancer le traitement batch"
    4. **Suivez l'avancement** dans le tableau des détails en temps réel
    5. **Analysez les résultats** dans les graphiques et tableaux

    ### Paramètres recommandés
    - **Workers** : 4 pour un bon équilibre performance/ressources
    - **Test** : Commencez avec une limite de 10-20 AOMs pour tester

    ### Indicateurs d'avancement
    - **🔄 En cours** : Traitement en cours avec étape détaillée
    - **✅ Succès** : Traitement terminé avec succès
    - **❌ Erreur** : Erreur rencontrée (voir colonne erreur)
    - **⚠️ Sans données** : Aucun contenu pertinent trouvé

    ### Gestion des erreurs
    - Les erreurs sont capturées et loggées
    - Vous pouvez relancer spécifiquement les AOMs en erreur ou sans données
    - Les résultats sont sauvegardés même en cas d'erreurs partielles
    - L'étape où l'erreur s'est produite est indiquée

    ### Suivi en temps réel
    Pendant le traitement, vous verrez :
    - Un tableau avec le statut de chaque AOM
    - L'étape courante de traitement
    - Une barre de progression visuelle
    - Des métriques en temps réel (en cours, terminés, erreurs)
    """
    )

# Nettoyage en cas de fermeture de la page
if st.session_state.get("processing_active", False):
    # Arrêter les threads en cas de rechargement de page
    st.session_state.processing_active = False
