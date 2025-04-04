import asyncio

import nest_asyncio
import streamlit as st

# Initialize the event loop before importing crawl4ai
# flake8: noqa: E402
nest_asyncio.apply()
from utils.crawler_utils import CrawlerManager
from utils.db_utils import load_urls_data_from_db

st.title("Extraction multipages sur les AOMs")


def toggle_crawling():
    if "is_crawling" not in st.session_state:
        st.session_state.is_crawling = False

    st.session_state.is_crawling = not st.session_state.is_crawling
    return st.session_state.is_crawling


# retrieve urls from db
urls_data = load_urls_data_from_db()
DATA = urls_data[["n_siren_aom", "nom_aom", "site_web_principal"]].to_dict(
    orient="records"
)
st.write(DATA)
# init crawler
if "crawler_manager" not in st.session_state:

    def reset_crawler_callback():
        st.session_state.crawler_manager = None

    st.session_state.crawler_manager = CrawlerManager(
        on_crawler_reset=reset_crawler_callback
    )
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

# Ajout des tags de mots-clés
default_keywords = [
    "boutique",
    "tarif",
    "abonnement",
    "ticket",
    "pass",
    "carte",
    "titre",
    "solidaire",
    "tarif solidaire",
]
if "available_keywords" not in st.session_state:
    st.session_state.available_keywords = default_keywords.copy()
if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = default_keywords.copy()

# Ajout d'un champ pour les mots-clés personnalisés
new_keyword = st.text_input(
    label="Ajouter un nouveau mot-clé :",
    placeholder="Entrez un nouveau mot-clé et appuyez sur Entrée",
    help="Le nouveau mot-clé sera ajouté à la liste des mots-clés disponibles",
)
if new_keyword:
    if (
        new_keyword not in st.session_state.available_keywords
        and new_keyword not in st.session_state.selected_keywords
    ):
        st.session_state.available_keywords.append(new_keyword)
        st.session_state.selected_keywords.append(new_keyword)
        st.rerun()

# Utilisation de multiselect pour les mots-clés
st.session_state.selected_keywords = st.multiselect(
    label="🏷️ Mots-clés pour la recherche :",
    options=st.session_state.available_keywords,
    default=st.session_state.selected_keywords,
    placeholder="Choisissez un ou plusieurs mots-clés",
    help="Sélectionnez les mots-clés qui seront utilisés pour "
    "la recherche dans les urls des pages web",
)

stop_button = st.button(
    "🛑 Arrêter l'extraction",
    help="Cliquez pour arrêter l'extraction en cours",
    disabled=not st.session_state.get("is_crawling", False),
    on_click=toggle_crawling,
)

start_button = st.button(
    "🕷️ Lancer l'extraction",
    help="Cliquez pour lancer l'extraction des données sur les sites web",
    disabled=st.session_state.get("is_crawling", False),
    on_click=toggle_crawling,
)

if start_button:
    with st.spinner("Extraction en cours..."):
        for data in DATA[:10]:
            url = data["site_web_principal"]
            n_siren_aom = data["n_siren_aom"]
            nom_aom = data["nom_aom"]
            with st.expander(f"N° SIREN AOM : {n_siren_aom}, {nom_aom}"):
                try:
                    loop = st.session_state.loop
                    asyncio.set_event_loop(loop)
                    pages = loop.run_until_complete(
                        st.session_state.crawler_manager.fetch_content(
                            url,
                            st.session_state.selected_keywords,
                        )
                    )
                    st.write(f"Nombre de pages : {len(pages)}")
                except Exception as e:
                    st.error(f"⚠️ Une erreur est survenue : {str(e)}")
