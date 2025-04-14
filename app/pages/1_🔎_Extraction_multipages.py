import asyncio

import nest_asyncio
import streamlit as st

# Initialize the event loop before importing crawl4ai
# flake8: noqa: E402
nest_asyncio.apply()
from datetime import datetime, timedelta
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from utils.crawler_utils import CrawlerManager
from utils.db_utils import get_postgres_cs, load_urls_data_from_db

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
    "offres",
    "boutique",
    "tarif",
    "abonnement",
    "solidaire",
    "acheter",
    "pass",
    "titre",
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

# Ajout d'un bouton pour vider la table tarification_raw
if st.button(
    "🗑️ Vider la base de données",
    help="Attention: Cette action supprimera toutes les données de la table tarification_raw",
    type="secondary",
):
    try:
        engine = create_engine(get_postgres_cs())
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE tarification_raw"))
            conn.commit()
        st.success("✅ La table tarification_raw a été vidée avec succès.")
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la suppression des données: {str(e)}")

# Boutons existants pour démarrer/arrêter l'extraction
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
        # Connexion à la base de données
        engine = create_engine(get_postgres_cs())

        for data in DATA[:10]:
            url = data["site_web_principal"]
            n_siren_aom = data["n_siren_aom"]
            nom_aom = data["nom_aom"]

            with st.expander(f"N° SIREN AOM : {n_siren_aom}, {nom_aom}"):
                try:
                    # Vérifier si l'URL a été scrapée récemment (moins d'1h)
                    domain = urlparse(url).netloc
                    query = text(
                        """
                        SELECT * FROM tarification_raw 
                        WHERE url_source LIKE :domain_pattern 
                        AND date_scraping > :cutoff_time
                    """
                    )

                    with engine.connect() as conn:
                        result = conn.execute(
                            query,
                            {
                                "domain_pattern": f"%{domain}%",
                                "cutoff_time": datetime.now()
                                - timedelta(hours=1),
                            },
                        ).fetchone()

                    if result:
                        st.info(
                            f"⏱️ URL {url} déjà scrapée récemment. Passage à l'URL suivante."
                        )
                        continue

                    # Scraper l'URL
                    loop = st.session_state.loop
                    asyncio.set_event_loop(loop)
                    pages = loop.run_until_complete(
                        st.session_state.crawler_manager.fetch_content(
                            url,
                            st.session_state.selected_keywords,
                        )
                    )

                    # Sauvegarder les données dans la base de données
                    for page in pages:
                        with engine.connect() as conn:
                            conn.execute(
                                text(
                                    """
                                INSERT INTO tarification_raw 
                                (n_siren_aom, url_source, url_page, contenu_scrape)
                                VALUES (:n_siren_aom, :url_source, :url_page, :contenu_scrape)
                            """
                                ),
                                {
                                    "n_siren_aom": n_siren_aom,
                                    "url_source": url,
                                    "url_page": page.url,
                                    "contenu_scrape": page.markdown,
                                },
                            )
                            conn.commit()

                    # Créer un onglet par page
                    tabs = st.tabs([f"Page {i+1}" for i in range(len(pages))])
                    for i, page in enumerate(pages):
                        with tabs[i]:
                            st.markdown(f"{page.url}")
                            st.markdown(page.markdown)

                except Exception as e:
                    st.error(f"⚠️ Une erreur est survenue : {str(e)}")
