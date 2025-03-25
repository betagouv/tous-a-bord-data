import asyncio

import nest_asyncio
import pandas as pd
import psycopg2
import streamlit as st
from crawl4ai import AsyncWebCrawler, BrowserConfig
from utils.db_utils import get_postgres_cs

asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()


async def scrape_transport_sites(keywords):
    # Create a state variable for stopping the scraping
    if "stop_scraping" not in st.session_state:
        st.session_state.stop_scraping = False
    conn = psycopg2.connect(get_postgres_cs())
    cur = conn.cursor()
    failed_urls = []
    # Check the total number of unique SIRENs
    cur.execute(
        """
        SELECT COUNT(DISTINCT n_siren_aom)
        FROM transport_offers
        WHERE n_siren_aom IS NOT NULL
    """
    )
    total_siren = cur.fetchone()[0]
    st.write(f"**Nombre total de SIREN uniques:** {total_siren}")
    if total_siren != 211:
        st.warning(
            f"Attention: {total_siren} SIREN trouvés au lieu des 211 attendus"
        )
    # Main query
    cur.execute(
        """
        SELECT DISTINCT t.n_siren_aom, t.site_web_principal, t.autorite
        FROM transport_offers t
        WHERE t.n_siren_aom IS NOT NULL
        AND t.site_web_principal IS NOT NULL
        ORDER BY t.autorite
    """
    )
    sites = cur.fetchall()
    progress_bar = st.progress(0)
    status_container = st.empty()
    browser_config = BrowserConfig(
        browser_type="chromium", headless=True, verbose=True
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for idx, (siren, url, nom_aom) in enumerate(sites):
            # Check if the stop has been requested
            if st.session_state.stop_scraping:
                st.warning("Scraping stopped by the user")
                break
            # Update the progress
            progress = (idx + 1) / len(sites)
            progress_bar.progress(progress)
            status_col1, status_col2, status_col3 = status_container.columns(
                [2, 2, 1]
            )
            with status_col1:
                st.write(f"**AOM:** {nom_aom}")
            with status_col2:
                st.write(f"**URL:** {url}")
            try:
                result = await crawler.arun(url=url, search_terms=keywords)
                if result.markdown and len(result.markdown.strip()) > 1:
                    cur.execute(
                        """
                        INSERT INTO tarification_raw
                        (n_siren_aom, url_source, contenu_scrape)
                        VALUES (%s, %s, %s)
                    """,
                        (siren, url, result.markdown),
                    )
                    conn.commit()
                    with status_col3:
                        st.success("✓")
                    with st.expander(f"Contenu scrapé pour {nom_aom}"):
                        st.markdown(result.markdown)
                else:
                    failed_urls.append((nom_aom, url))
                    with status_col3:
                        st.error("✗")
            except Exception as e:
                failed_urls.append((nom_aom, url))
                with status_col3:
                    st.error("✗")
                st.error(f"Erreur pour {url}: {str(e)}")
                continue
    # Initialize the stop flag
    st.session_state.stop_scraping = False
    cur.close()
    conn.close()
    return failed_urls


def run_scraping(keywords):
    try:
        loop = asyncio.get_event_loop()
        failed_urls = loop.run_until_complete(scrape_transport_sites(keywords))
        if failed_urls:
            st.error("Sites ayant échoué au scraping:")
            for nom_aom, url in failed_urls:
                st.write(f"- **{nom_aom}:** {url}")
    except Exception as e:
        st.error(f"Erreur pendant le scraping: {str(e)}")


def search_donnees_aom(terme_recherche):
    conn = psycopg2.connect(get_postgres_cs())
    cur = conn.cursor()  # Search by SIREN or AOM name
    cur.execute(
        """
        SELECT t.n_siren_aom, t.url_source, t.contenu_scrape,
        t.date_scraping, a.nom_aom
        FROM tarification_raw t
        LEFT JOIN aoms a ON t.n_siren_aom = a.n_siren_aom
        WHERE t.n_siren_aom ILIKE %s OR a.nom_aom ILIKE %s
        ORDER BY t.date_scraping DESC
    """,
        (f"%{terme_recherche}%", f"%{terme_recherche}%"),
    )
    resultats = cur.fetchall()
    cur.close()
    conn.close()
    return resultats


def check_empty_content():
    conn = psycopg2.connect(get_postgres_cs())
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.n_siren_aom,
               a.nom_aom,
               t.url_source,
               t.date_scraping,
               LENGTH(TRIM(t.contenu_scrape)) as taille_contenu
        FROM tarification_raw t
        LEFT JOIN aoms a ON t.n_siren_aom = a.n_siren_aom
        WHERE TRIM(t.contenu_scrape) = ''
           OR t.contenu_scrape IS NULL
           OR LENGTH(TRIM(t.contenu_scrape)) <= 1
        ORDER BY t.date_scraping DESC
    """
    )
    resultats = cur.fetchall()
    # Get the total number of scraped AOMs
    cur.execute("SELECT COUNT(DISTINCT n_siren_aom) FROM tarification_raw")
    total_aoms = cur.fetchone()[0]
    cur.close()
    conn.close()
    return resultats, total_aoms


st.title("Scraping et exploration des données tarifaires")

tab1, tab2, tab3 = st.tabs(["Scraping", "Diagnostic", "Exploration"])

with tab1:
    st.subheader("Scraping intégral")
    # Keywords input
    default_keywords = [
        "tarif",
        "tarification",
        "abonnement",
        "ticket",
        "solidaire",
        "social",
        "reduction",
        "transport",
    ]
    keywords_input = st.text_area(
        "Mots clés pour le scraping (un par ligne)",
        value="\n".join(default_keywords),
    )
    # Create two columns for the buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        start_button = st.button("Lancer le scraping")
    with col2:
        if st.button("Arrêter le scraping"):
            st.session_state.stop_scraping = True
            st.warning("Arrêt du scraping demandé...")
    if start_button:
        keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]
        with st.spinner("Scraping en cours..."):
            run_scraping(keywords)
            if not st.session_state.stop_scraping:
                st.success("Scraping terminé!")

with tab2:
    st.subheader("Diagnostic des contenus vides")
    resultats_vides, total_aoms = check_empty_content()
    st.write(f"**Nombre total d'AOMs scrapées:** {total_aoms}")
    st.write(f"**Nombre d'AOMs avec contenu vide:** {len(resultats_vides)}")
    if resultats_vides:
        st.write("**Liste des AOMs à rescraper:**")
        df_vides = pd.DataFrame(
            resultats_vides,
            columns=[
                "SIREN",
                "Nom AOM",
                "URL",
                "Date scraping",
                "Taille contenu",
            ],
        )
        st.dataframe(df_vides)
        # Option to rescrape in bulk
        if st.button("Rescrape all empty AOMs"):
            with st.spinner("Rescraping in progress..."):
                for siren, _, _, _, _ in resultats_vides:
                    run_scraping(siren)
                st.success("Rescraping terminé!")

with tab3:
    st.subheader("Exploration des données")
    terme_recherche = st.text_input("Rechercher par nom d'AOM ou numéro SIREN")

    # Display search results
    if terme_recherche:
        resultats = search_donnees_aom(terme_recherche)
        if resultats:
            for siren, url, contenu_scrape, date, nom_aom in resultats:
                st.write(f"**Nom de l'AOM:** {nom_aom}")
                st.write(f"**SIREN de l'AOM:** {siren}")
                st.write(f"**URL source:** {url}")
                st.write(f"**Date de scraping:** {date.strftime('%d/%m/%Y')}")
                with st.expander("**Contenu du scraping**"):
                    st.markdown(contenu_scrape)
        else:
            st.info("Aucun résultat trouvé pour cette recherche.")
