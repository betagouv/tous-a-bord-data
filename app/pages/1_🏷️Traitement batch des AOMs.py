import asyncio

import streamlit as st
from services.batch_processor import BatchProcessor
from services.llm_services import LLM_MODELS
from sqlalchemy import create_engine, text
from utils.db_utils import get_postgres_cs

st.set_page_config(
    page_title="Traitement batch des AOMs",
    page_icon="🚀",
)

st.title("Traitement batch des AOMs")

# Initialisation de la connexion à la base de données
engine = create_engine(get_postgres_cs())

# Récupération de la liste des AOMs
with engine.connect() as conn:
    aoms = conn.execute(
        text(
            """
            SELECT DISTINCT
                t.n_siren_aom,
                a.nom_aom,
                COUNT(DISTINCT t.url_source) as nb_sources,
                STRING_AGG(DISTINCT t.url_source, ' | ') as sources,
                STRING_AGG(DISTINCT t.contenu_scrape, '\n\n') as contenus
            FROM tarification_raw t
            LEFT JOIN aoms a ON t.n_siren_aom = a.n_siren_aom
            GROUP BY t.n_siren_aom, a.nom_aom
            ORDER BY COUNT(DISTINCT t.url_source) DESC, a.nom_aom
            """
        )
    ).fetchall()

st.write(f"Nombre total d'AOMs disponibles : {len(aoms)}")

# Sélection des AOMs
st.write("### 1. Sélection des AOMs")
col1, col2 = st.columns(2)

with col1:
    select_all = st.checkbox("Sélectionner toutes les AOMs", value=False)

with col2:
    if select_all:
        selected_aoms = [aom[0] for aom in aoms]
        st.info(f"✅ {len(selected_aoms)} AOMs sélectionnées")
    else:

        def format_aom_option(x):
            nom = next((a[1] for a in aoms if a[0] == x), "Unknown")
            sources = next((a[2] for a in aoms if a[0] == x), 0)
            return f"{x} - {nom} ({sources} sources)"

        selected_aoms = st.multiselect(
            "Sélectionner les AOMs à traiter",
            options=[aom[0] for aom in aoms],
            format_func=format_aom_option,
            default=[
                aom[0] for aom in aoms[:5]
            ],  # 5 premières AOMs par défaut
        )
        if selected_aoms:
            st.info(f"✅ {len(selected_aoms)} AOMs sélectionnées")
        else:
            st.warning("⚠️ Aucune AOM sélectionnée")
            st.stop()

# Configuration du traitement batch
st.write("### 2. Sélection des étapes")
steps = {
    "filtering": "Filtrage du contenu",
    "pre_formatting": "Pré-formatage en langage naturel",
    "tags_formatting": "Formatage en tags",
}

selected_steps = []
for step_id, step_name in steps.items():
    if st.checkbox(step_name, key=f"step_{step_id}"):
        selected_steps.append(step_id)

if not selected_steps:
    st.warning("⚠️ Veuillez sélectionner au moins une étape")
    st.stop()

# Configuration des méthodes pour chaque étape
st.write("### 3. Configuration des méthodes")

methods_config = {}

# Configuration du filtrage si sélectionné
if "filtering" in selected_steps:
    with st.expander("🎯 Configuration du filtrage"):
        filter_method = st.radio(
            "Méthode de filtrage",
            options=["NLP (SpaCy)", "LLM"],
            key="filter_method",
        )
        if filter_method == "LLM":
            filter_model = st.selectbox(
                "Modèle LLM pour le filtrage",
                options=list(LLM_MODELS.keys()),
                key="filter_model",
            )
        methods_config["filtering"] = {
            "method": filter_method,
            "model": filter_model if filter_method == "LLM" else None,
        }

# Configuration du pré-formatage si sélectionné
if "pre_formatting" in selected_steps:
    with st.expander("🧹 Configuration du pré-formatage"):
        preformat_model = st.selectbox(
            "Modèle LLM pour le pré-formatage",
            options=list(LLM_MODELS.keys()),
            key="preformat_model",
        )
        methods_config["pre_formatting"] = {"model": preformat_model}

# Configuration du formatage YAML si sélectionné
if "yaml_formatting" in selected_steps:
    with st.expander("📖 Configuration du formatage YAML"):
        yaml_model = st.selectbox(
            "Modèle LLM pour le formatage YAML",
            options=list(LLM_MODELS.keys()),
            key="yaml_model",
        )
        methods_config["yaml_formatting"] = {"model": yaml_model}

# Configuration générale
st.write("### 4. Configuration générale")
max_concurrent = st.slider(
    "Nombre maximum d'AOMs en parallèle",
    min_value=1,
    max_value=10,
    value=3,
    help="Attention : une valeur trop élevée peut surcharger le système",
)

# Préparation des données des AOMs sélectionnées
aoms_data = []
for aom in aoms:
    if aom[0] in selected_aoms:
        aoms_data.append(
            {
                "siren": aom[0],
                "nom": aom[1],
                "content": aom[4],  # Contenu déjà scrapé
            }
        )

# Création des conteneurs pour le suivi de la progression
progress_containers = {}
for aom in aoms_data:
    with st.expander(f"🔄 {aom['nom']} ({aom['siren']})"):
        progress_containers[aom["siren"]] = {
            "status": st.empty(),
            "progress": st.progress(0),
        }


def update_progress(siren: str, stage: str, progress: float):
    """Met à jour la progression pour une AOM donnée"""
    if siren in progress_containers:
        progress_containers[siren]["status"].write(
            f"Étape en cours : {steps.get(stage, stage)}"
        )
        progress_containers[siren]["progress"].progress(progress)


# Bouton pour lancer le traitement
if st.button("🚀 Lancer le traitement batch"):
    try:
        # Initialisation du processeur batch
        processor = BatchProcessor(
            max_concurrent_aoms=max_concurrent,
            selected_steps=selected_steps,
            methods_config=methods_config,
        )

        # Fonction de callback pour la progression
        def progress_callback(siren, stage, progress):
            update_progress(siren, stage, progress)

        # Lancement du traitement asynchrone
        results = asyncio.run(
            processor.process_all_aoms(
                aoms_data,
                progress_callback=progress_callback,
            )
        )

        # Affichage des résultats
        st.subheader("Résultats du traitement")

        success_count = len([r for r in results if "error" not in r])
        error_count = len([r for r in results if "error" in r])

        st.write(f"✅ {success_count} AOMs traitées avec succès")
        if error_count > 0:
            st.write(f"❌ {error_count} AOMs en erreur")

        # Affichage détaillé des résultats
        for result in results:
            title = f"📊 Résultats pour {result['nom']} ({result['siren']})"
            with st.expander(title):
                if "error" in result:
                    st.error(f"Erreur : {result['error']}")
                else:
                    for step in selected_steps:
                        if step in result:
                            st.write(f"Résultat de l'étape {steps[step]} :")
                            if step == "yaml_formatting":
                                st.code(result[step], language="yaml")
                            else:
                                st.text_area(
                                    "",
                                    value=result[step],
                                    height=200,
                                    disabled=True,
                                )

    except Exception as e:
        error_msg = (
            f"Une erreur est survenue lors du traitement batch : {str(e)}"
        )
        st.error(error_msg)
        st.error("Détails de l'erreur :", str(e.__class__.__name__))
