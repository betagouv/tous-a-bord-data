import pandas as pd
import streamlit as st
from services.evaluation_service import evaluation_service

st.set_page_config(
    page_title="Évaluation HITL - Transport Tarifs",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Évaluation Human-in-the-Loop (HITL)")
st.markdown("Interface d'évaluation pour améliorer la qualité du pipeline RAG")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
mode = st.sidebar.selectbox(
    "Mode d'évaluation",
    [
        "📈 Tableau de bord",
        "🔍 Évaluer les runs",
        "📋 Historique des évaluations",
    ],
)

if mode == "📈 Tableau de bord":
    st.header("Tableau de bord des évaluations")

    # Récupérer les statistiques
    with st.spinner("Chargement des statistiques..."):
        stats = evaluation_service.get_evaluation_stats()

    # Afficher les métriques principales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total des runs",
            stats["total_runs"],
            help="Nombre total de runs LLM exécutés",
        )

    with col2:
        st.metric(
            "Runs évalués",
            stats["runs_with_feedback"],
            help="Nombre de runs ayant reçu un feedback humain",
        )

    with col3:
        coverage_pct = f"{stats['feedback_coverage']:.1%}"
        st.metric(
            "Couverture d'évaluation",
            coverage_pct,
            help="Pourcentage de runs évalués par un humain",
        )

    with col4:
        avg_score = f"{stats['average_score']:.2f}"
        st.metric(
            "Score moyen", avg_score, help="Score moyen des évaluations (0-1)"
        )

    # Graphique de progression (placeholder pour l'instant)
    st.subheader("Évolution de la qualité")
    st.info(
        "📊 Graphiques de progression à venir - nécessite plus de "
        "données historiques"
    )

    # Recommandations
    st.subheader("Recommandations")
    if stats["feedback_coverage"] < 0.2:
        st.warning(
            "⚠️ Couverture d'évaluation faible. Évaluez plus de runs "
            "pour améliorer le système."
        )
    elif stats["average_score"] < 0.7:
        st.warning(
            "⚠️ Score moyen faible. Analysez les runs mal notés pour "
            "identifier les problèmes."
        )
    else:
        st.success(
            "✅ Bon niveau d'évaluation. Continuez à évaluer " "régulièrement."
        )

elif mode == "🔍 Évaluer les runs":
    st.header("Évaluation des runs LLM")

    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input(
            "Nombre de runs à afficher", min_value=10, max_value=100, value=20
        )
    with col2:
        show_only_unevaluated = st.checkbox(
            "Afficher seulement les runs non évalués", value=True
        )

    # Récupérer les runs
    with st.spinner("Chargement des runs..."):
        runs = evaluation_service.list_recent_runs(limit=limit)

    if not runs:
        st.warning(
            "Aucun run trouvé. Assurez-vous que LangSmith est "
            "correctement configuré."
        )
        st.stop()

    # Filtrer les runs non évalués si demandé
    if show_only_unevaluated:
        unevaluated_runs = []
        for run in runs:
            feedbacks = evaluation_service.get_feedback_for_run(run.id)
            if not feedbacks:
                unevaluated_runs.append(run)
        runs = unevaluated_runs

    if not runs:
        st.info("✅ Tous les runs récents ont été évalués !")
        st.stop()

    st.write(f"**{len(runs)} runs** à évaluer")

    # Interface d'évaluation pour chaque run
    for i, run in enumerate(runs):
        run_title = (
            f"Run {i+1}: {run.name or 'Sans nom'} - "
            f"{run.start_time.strftime('%Y-%m-%d %H:%M')}"
        )
        with st.expander(run_title):

            # Informations du run
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**ID:** `{run.id}`")
                model_name = run.extra.get("metadata", {}).get(
                    "model", "Non spécifié"
                )
                st.write(f"**Modèle:** {model_name}")
                st.write(
                    f"**Durée:** {run.execution_order}ms"
                    if run.execution_order
                    else "N/A"
                )

            with col2:
                st.write(f"**Statut:** {run.status}")
                st.write(f"**Type:** {run.run_type}")
                if run.error:
                    st.error(f"**Erreur:** {run.error}")

            # Inputs et outputs
            if run.inputs:
                st.subheader("📥 Entrées")
                for key, value in run.inputs.items():
                    if isinstance(value, str) and len(value) > 500:
                        st.text_area(
                            f"**{key}:**",
                            value[:500] + "...",
                            height=100,
                            disabled=True,
                        )
                    else:
                        st.write(f"**{key}:** {value}")

            if run.outputs:
                st.subheader("📤 Sorties")
                for key, value in run.outputs.items():
                    if isinstance(value, str):
                        st.text_area(
                            f"**{key}:**", value, height=150, disabled=True
                        )
                    else:
                        st.write(f"**{key}:** {value}")

            # Interface d'évaluation
            st.subheader("📊 Évaluation")

            eval_col1, eval_col2, eval_col3 = st.columns(3)

            with eval_col1:
                quality_score = st.select_slider(
                    "Qualité générale",
                    options=[0, 0.25, 0.5, 0.75, 1.0],
                    format_func=lambda x: {
                        0: "Très mauvais",
                        0.25: "Mauvais",
                        0.5: "Moyen",
                        0.75: "Bon",
                        1.0: "Excellent",
                    }[x],
                    key=f"quality_{run.id}",
                )

            with eval_col2:
                relevance_score = st.select_slider(
                    "Pertinence",
                    options=[0, 0.25, 0.5, 0.75, 1.0],
                    format_func=lambda x: {
                        0: "Non pertinent",
                        0.25: "Peu pertinent",
                        0.5: "Moyennement pertinent",
                        0.75: "Pertinent",
                        1.0: "Très pertinent",
                    }[x],
                    key=f"relevance_{run.id}",
                )

            with eval_col3:
                accuracy_score = st.select_slider(
                    "Précision",
                    options=[0, 0.25, 0.5, 0.75, 1.0],
                    format_func=lambda x: {
                        0: "Très imprécis",
                        0.25: "Imprécis",
                        0.5: "Moyennement précis",
                        0.75: "Précis",
                        1.0: "Très précis",
                    }[x],
                    key=f"accuracy_{run.id}",
                )

            # Commentaires et corrections
            comment = st.text_area(
                "Commentaires (optionnel)",
                placeholder=(
                    "Décrivez les problèmes identifiés ou les "
                    "améliorations possibles..."
                ),
                key=f"comment_{run.id}",
            )

            correction = st.text_area(
                "Correction proposée (optionnel)",
                placeholder="Proposez une version corrigée de la sortie...",
                key=f"correction_{run.id}",
            )

            # Bouton de soumission
            if st.button("💾 Sauvegarder l'évaluation", key=f"submit_{run.id}"):
                with st.spinner("Sauvegarde en cours..."):
                    # Créer les feedbacks
                    feedback_ids = []

                    # Feedback qualité
                    if quality_score is not None:
                        fid = evaluation_service.create_feedback(
                            run_id=run.id,
                            key="quality",
                            score=quality_score,
                            comment=comment if comment else None,
                        )
                        if fid:
                            feedback_ids.append(fid)

                    # Feedback pertinence
                    if relevance_score is not None:
                        fid = evaluation_service.create_feedback(
                            run_id=run.id,
                            key="relevance",
                            score=relevance_score,
                            comment=comment if comment else None,
                        )
                        if fid:
                            feedback_ids.append(fid)

                    # Feedback précision
                    if accuracy_score is not None:
                        fid = evaluation_service.create_feedback(
                            run_id=run.id,
                            key="accuracy",
                            score=accuracy_score,
                            comment=comment if comment else None,
                            correction={"corrected_output": correction}
                            if correction
                            else None,
                        )
                        if fid:
                            feedback_ids.append(fid)

                    if feedback_ids:
                        success_msg = (
                            f"✅ Évaluation sauvegardée ! "
                            f"({len(feedback_ids)} feedbacks créés)"
                        )
                        st.success(success_msg)
                        st.rerun()
                    else:
                        st.error("❌ Erreur lors de la sauvegarde")

elif mode == "📋 Historique des évaluations":
    st.header("Historique des évaluations")

    # Récupérer les runs avec feedback
    with st.spinner("Chargement de l'historique..."):
        runs = evaluation_service.list_recent_runs(limit=50)

        # Créer un tableau avec les évaluations
        evaluation_data = []
        for run in runs:
            feedbacks = evaluation_service.get_feedback_for_run(run.id)
            if feedbacks:
                for feedback in feedbacks:
                    evaluation_data.append(
                        {
                            "Date": run.start_time.strftime("%Y-%m-%d %H:%M"),
                            "Run ID": run.id[:8] + "...",
                            "Nom": run.name or "Sans nom",
                            "Type d'évaluation": feedback.key,
                            "Score": feedback.score,
                            "Commentaire": feedback.comment[:100] + "..."
                            if feedback.comment and len(feedback.comment) > 100
                            else feedback.comment,
                            "Correction": "Oui"
                            if feedback.correction
                            else "Non",
                        }
                    )

    if evaluation_data:
        df = pd.DataFrame(evaluation_data)

        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            eval_types = st.multiselect(
                "Types d'évaluation",
                options=df["Type d'évaluation"].unique(),
                default=df["Type d'évaluation"].unique(),
            )

        with col2:
            min_score = st.slider("Score minimum", 0.0, 1.0, 0.0, 0.25)

        # Filtrer le dataframe
        filtered_df = df[
            (df["Type d'évaluation"].isin(eval_types))
            & (df["Score"] >= min_score)
        ]

        # Afficher le tableau
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        # Statistiques rapides
        if not filtered_df.empty:
            st.subheader("Statistiques")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Nombre d'évaluations", len(filtered_df))

            with col2:
                avg_score = filtered_df["Score"].mean()
                st.metric("Score moyen", f"{avg_score:.2f}")

            with col3:
                corrections_count = len(
                    filtered_df[filtered_df["Correction"] == "Oui"]
                )
                st.metric("Corrections proposées", corrections_count)

    else:
        st.info(
            "Aucune évaluation trouvée. Commencez par évaluer "
            "quelques runs !"
        )

# Footer avec informations de configuration
st.sidebar.markdown("---")
st.sidebar.subheader("Configuration LangSmith")
if evaluation_service.client:
    st.sidebar.success("✅ Connecté à LangSmith")
    st.sidebar.write(f"**Projet:** {evaluation_service.project_name}")
else:
    st.sidebar.error("❌ Erreur de connexion LangSmith")
    st.sidebar.write("Vérifiez vos variables d'environnement :")
    st.sidebar.code(
        """
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=transport-tarifs-pipeline
    """
    )
