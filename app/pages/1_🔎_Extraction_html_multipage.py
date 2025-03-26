import asyncio

import nest_asyncio
import pandas as pd
import psycopg2
import streamlit as st
from utils.db_utils import get_postgres_cs

asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

# flake8: noqa: E402
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    ContentTypeFilter,
    FilterChain,
    URLPatternFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer


def clear_tarification_data():
    conn = psycopg2.connect(get_postgres_cs())
    cur = conn.cursor()
    try:
        #  Emptying the table
        cur.execute("TRUNCATE TABLE tarification_raw;")
        # Resetting the ID sequence
        cur.execute("ALTER SEQUENCE tarification_raw_id_seq RESTART WITH 1;")
        conn.commit()
        st.success("Table tarification_raw emptied with success!")
    except Exception as e:
        st.error(f"Erreur lors du nettoyage de la table: {str(e)}")
    finally:
        cur.close()
        conn.close()


# Initialize session state variables
if "stop_scraping" not in st.session_state:
    st.session_state.stop_scraping = False


async def scrape_transport_sites(
    keywords,
    progress_callback=None,
    url_patterns=None,
    max_pages_per_site=10,
    score_threshold=0.1,
    crawl_strategy="bestfirst",
):
    conn = psycopg2.connect(get_postgres_cs())
    cur = conn.cursor()
    failed_urls = []

    # Vider la table au début
    try:
        cur.execute("TRUNCATE TABLE tarification_raw;")
        conn.commit()
    except Exception as e:
        st.error(f"Erreur lors du nettoyage de la table: {str(e)}")
        cur.close()
        conn.close()
        return [], []

    cur.execute(
        """
        SELECT t.n_siren_aom, t.site_web_principal, t.autorite
        FROM transport_offers t
        WHERE t.n_siren_aom IS NOT NULL
        AND t.site_web_principal IS NOT NULL
        ORDER BY t.autorite
        """
    )
    sites = cur.fetchall()

    browser_config = BrowserConfig(verbose=True)
    # Configuration du scorer
    scorer = KeywordRelevanceScorer(keywords=keywords, weight=0.7)

    # Configuration de la stratégie de crawling
    strategy = BestFirstCrawlingStrategy(
        max_depth=3,
        include_external=False,
        max_pages=max_pages_per_site,
        url_scorer=scorer,
    )

    # Configuration complète
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        word_count_threshold=5,
        exclude_external_links=True,
        remove_overlay_elements=True,
        process_iframes=True,
        verbose=True,
        stream=False,
    )

    results_list = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for idx, (siren, url, nom_aom) in enumerate(sites):
            if st.session_state.stop_scraping:
                break

            success = False
            error = None
            visited_urls_content = {}

            st.write(
                f"Démarrage du crawl pour {nom_aom} ({idx+1}/{len(sites)}) - Limite de {max_pages_per_site} pages"
            )

            try:
                # Ajout d'un délai entre chaque site
                if idx > 0:
                    await asyncio.sleep(2)

                results = await crawler.arun(url=url, config=config)
                if results and isinstance(results, list):
                    st.write(f"Pages trouvées pour {nom_aom}: {len(results)}")
                    for result in results:
                        if result.success:
                            visited_urls_content[result.url] = result.markdown
                            cur.execute(
                                """
                                INSERT INTO tarification_raw
                                (n_siren_aom, url_source, url_page, contenu_scrape)
                                VALUES (%s, %s, %s, %s)
                            """,
                                (siren, url, result.url, result.markdown),
                            )

                    conn.commit()
                    success = bool(visited_urls_content)
                else:
                    st.write(f"Aucun résultat pour {nom_aom}")

            except Exception as e:
                error = str(e)
                st.write(f"Erreur pour {nom_aom}: {str(e)}")
                failed_urls.append((nom_aom, url))

            # Mise à jour des résultats
            current_result = {
                "siren": siren,
                "url": url,
                "nom_aom": nom_aom,
                "success": success,
                "visited_urls": visited_urls_content,
                "error": error,
            }
            results_list.append(current_result)

            if progress_callback:
                progress_callback(idx, len(sites), current_result)

            st.write(f"Fin du traitement pour {nom_aom}")
            st.write("---")

    cur.close()
    conn.close()
    return results_list, failed_urls


def streamlit_progress_callback(idx, total, result):
    """Callback d'affichage pour Streamlit"""
    progress = (idx + 1) / total
    st.session_state.progress_bar.progress(progress)

    # Affichage principal de l'AOM
    with st.expander(f"**{result['nom_aom']}** - {result['url']}"):
        # Status de succès/échec
        if result["success"]:
            st.success("✓ Scraping réussi")
        else:
            st.error("✗ Échec du scraping")

        if result["success"] and result.get("visited_urls"):
            st.write(
                f"**Nombre de pages trouvées:** {len(result['visited_urls'])}"
            )

            # Créer des tabs pour chaque URL visitée
            tabs = st.tabs(
                [f"Page {i+1}" for i in range(len(result["visited_urls"]))]
            )

            # Remplir chaque tab avec le contenu correspondant
            for i, (visited_url, page_content) in enumerate(
                result["visited_urls"].items()
            ):
                with tabs[i]:
                    st.markdown(
                        f"**URL source:** [{result['url']}]({result['url']})"
                    )
                    st.markdown(
                        f"**URL visitée:** [{visited_url}]({visited_url})"
                    )
                    st.markdown("---")
                    st.markdown(page_content)

        elif result["error"]:
            st.error(f"Erreur: {result['error']}")


def display_scraping_results(results):
    """Fonction d'affichage pure"""
    progress_bar = st.progress(0)
    status_container = st.empty()

    for idx, result in enumerate(results):
        progress = (idx + 1) / len(results)
        progress_bar.progress(progress)

        status_col1, status_col2, status_col3 = status_container.columns(
            [2, 2, 1]
        )
        with status_col1:
            st.write(f"**AOM:** {result['nom_aom']}")
        with status_col2:
            st.markdown(f"[{result['url']}]({result['url']})")
        with status_col3:
            if result["success"]:
                st.success("✓")
            else:
                st.error("✗")

        if result["success"]:
            with st.expander(f"Contenu scrapé pour {result['nom_aom']}"):
                st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                st.markdown("---")
                st.markdown(result["content"])
        elif result["error"]:
            st.error(f"Erreur: {result['error']}")


def run_scraping(keywords):
    """Wrapper pour exécuter le scraping asynchrone"""
    try:
        loop = asyncio.get_event_loop()
        results, failed_urls = loop.run_until_complete(
            scrape_transport_sites(keywords)
        )
        return results, failed_urls
    except Exception as e:
        st.error(f"Erreur pendant le scraping: {str(e)}")
        return [], []


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
        SELECT t.n_siren_aom as siren,
               o.nom_aom as nom_aom,
               t.url_source as url,
               t.date_scraping as date_scraping,
               LENGTH(TRIM(t.contenu_scrape)) as taille_contenu
        FROM tarification_raw t
        LEFT JOIN aoms o ON t.n_siren_aom = o.n_siren_aom
        WHERE t.n_siren_aom IS NOT NULL
        AND (
            TRIM(t.contenu_scrape) = ''
            OR t.contenu_scrape IS NULL
            OR LENGTH(TRIM(t.contenu_scrape)) <= 1
        )
        ORDER BY t.date_scraping DESC
    """
    )
    resultats = cur.fetchall()

    # Get the total number of scraped AOMs
    cur.execute(
        """
        SELECT COUNT(DISTINCT n_siren_aom) 
        FROM tarification_raw 
        WHERE n_siren_aom IS NOT NULL
    """
    )
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
        "solidaire",
        "social",
        "reduction",
    ]
    keywords_input = st.text_area(
        "Mots clés pour le scraping (un par ligne)",
        value="\n".join(default_keywords),
    )

    # URL patterns input
    default_patterns = [
        "tarif",
        "abonnement",
        "prix",
    ]
    patterns_input = st.text_area(
        "Patterns d'URLs à explorer (un par ligne)",
        value="\n".join(default_patterns),
    )

    # Paramètres de crawling
    col1, col2 = st.columns(2)
    with col1:
        max_pages_per_site = st.number_input(
            "Nombre maximum de pages à explorer par site",
            min_value=1,
            value=10,
        )
        score_threshold = st.slider(
            "Score minimum de pertinence",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
        )
    with col2:
        crawl_strategy = st.selectbox(
            "Stratégie d'exploration",
            options=["bestfirst"],
            help="bestfirst: exploration en profondeur",
        )

    # Boutons de contrôle
    col1, col2 = st.columns([4, 1])
    with col1:
        start_button = st.button("🚀 Lancer le scraping")
    with col2:
        stop_button = st.button("🛑 Arrêter")
        if stop_button:
            st.session_state.stop_scraping = True
            st.warning("Arrêt du scraping demandé...")

    if start_button:
        # Réinitialiser l'état d'arrêt
        st.session_state.stop_scraping = False

        keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]
        patterns = [p.strip() for p in patterns_input.split("\n") if p.strip()]

        # Créer les éléments d'interface et les stocker dans session_state
        st.session_state.progress_bar = st.progress(0)
        st.session_state.status_container = st.empty()

        with st.spinner("Scraping en cours..."):
            try:
                loop = asyncio.get_event_loop()
                results, failed_urls = loop.run_until_complete(
                    scrape_transport_sites(
                        keywords=keywords,
                        progress_callback=streamlit_progress_callback,
                        max_pages_per_site=max_pages_per_site,
                        score_threshold=score_threshold,
                        crawl_strategy=crawl_strategy,
                    )
                )

                if failed_urls:
                    st.error("Sites ayant échoué au scraping:")
                    for nom, url in failed_urls:
                        st.markdown(f"- **{nom}:** [{url}]({url})")

                if not st.session_state.stop_scraping:
                    st.success("Scraping terminé!")

            except Exception as e:
                st.error(f"Erreur pendant le scraping: {str(e)}")

        # Après le scraping, réinitialiser l'état
        st.session_state.stop_scraping = False

        if not st.session_state.stop_scraping:
            st.success("Scraping terminé!")
        else:
            st.info("Scraping arrêté par l'utilisateur")

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
                st.write(f"**URL source:** {url}")
                st.write(f"**Date de scraping:** {date.strftime('%d/%m/%Y')}")
                st.write(f"**SIREN de l'AOM:** {siren}")
                with st.expander("**Contenu du scraping**"):
                    st.markdown(contenu_scrape)
        else:
            st.info("Aucun résultat trouvé pour cette recherche.")
