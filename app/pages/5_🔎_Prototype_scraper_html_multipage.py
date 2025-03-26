import asyncio

import nest_asyncio
import streamlit as st

# flake8: noqa: E402
asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    ContentRelevanceFilter,
    FilterChain,
    SEOFilter,
    URLPatternFilter,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from utils.dataframe_utils import filter_dataframe
from utils.db_utils import load_urls_data_from_db


async def scraper_multipage(url, keywords):

    browser_config = BrowserConfig(verbose=True)

    url_filter = URLPatternFilter(
        patterns=[
            "*boutique*",
            "*tarif*",
            "*abonnement*",
            "*ticket*",
            "*pass*",
            "*carte*",
            "*titre*",
        ]
    )
    relevance_filter = ContentRelevanceFilter(
        query=" ".join(keywords), threshold=0.5
    )
    seo_filter = SEOFilter(threshold=0.5, keywords=keywords)

    scorer = KeywordRelevanceScorer(keywords=keywords, weight=1)

    strategy = BestFirstCrawlingStrategy(
        max_depth=2,
        max_pages=10,
        include_external=False,
        url_scorer=scorer,
        filter_chain=FilterChain(
            [
                url_filter,
                # relevance_filter,
                # seo_filter
            ]
        ),
    )

    run_config = CrawlerRunConfig(
        # Content filtering
        word_count_threshold=10,
        exclude_external_links=True,
        # Content processing
        remove_overlay_elements=True,  # Remove popups/modals
        process_iframes=True,  # Process iframe content
        # Deep crawling
        deep_crawl_strategy=strategy,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        return result


st.title("Prototype du scraper multipage")

aoms_urls_data = load_urls_data_from_db()

# Barre de recherche
search_term = st.text_input(
    "🔍 Rechercher une AOM",
    placeholder="Exemple : Brest Métropole",
)

# Filtrer les données
filtered_df = filter_dataframe(aoms_urls_data, search_term)

# Afficher le nombre de résultats
nb_results = len(filtered_df)
first = f"{nb_results} résultat{'s' if nb_results > 1 else ''}"
second = f"trouvé{'s' if nb_results > 1 else ''}"

st.write(f"📊{first} {second}")


# Afficher le tableau filtré

st.dataframe(
    filtered_df,
    column_config={
        "n_siren_aom": "SIREN AOM",
        "nom_aom": "Nom de l'AOM",
        "site_web_principal": "URL tarification",
        "nom_commercial": "Nom commercial",
        "exploitant": "Exploitant",
        "type_de_contrat": "Type de contrat",
        "population_aom": "Population de l'AOM",
        "nombre_membre_aom": "Nombre de membres de l'AOM",
        "surface_km_2": "Surface de l'AOM",
        "type_d_usagers_faibles_revenus": "Type d'usagers faibles revenus",
        "type_d_usagers_recherche_d_emplois": "Type d'usagers recherche d'emplois",
    },
    hide_index=True,
    use_container_width=True,
)

# Sélection d'une AOM via une liste déroulante
st.subheader("Sélectionner une url pour l'extraction")
url_options = filtered_df["site_web_principal"].tolist()
selected_url = st.selectbox("Choisir une url", url_options)


# Récupérer les informations de l'AOM sélectionnée
if selected_url:
    selected_url = filtered_df[
        filtered_df["site_web_principal"] == selected_url
    ].iloc[0]
    st.write(f"### URL sélectionnée : {selected_url['site_web_principal']}")
    url = selected_url["site_web_principal"]
    if not url.startswith(("http://", "https://")):
        st.error("L'URL doit commencer par http:// ou https://")
        st.stop()
    keywords_input = st.text_input(
        "Mots-clés (séparés par des virgules)",
        placeholder="Exemple : tarif, ticket, abonnement",
        help="Entrez les mots-clés qui vous intéressent pour filtrer les résultats",
    )
    scrape_button = st.button(
        "🔍 Extraire les critères", use_container_width=True
    )


if scrape_button:
    if not keywords_input.strip():
        st.warning("Veuillez entrer au moins un mot-clé")
        st.stop()

    # Convertir la chaîne de mots-clés en liste
    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    with st.spinner(
        "Extraction en cours... " "Cela peut prendre quelques instants."
    ):
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(scraper_multipage(url, keywords))

            # Créer un onglet par page
            tabs = st.tabs([f"Page {i+1}" for i in range(len(results))])

            for i, page in enumerate(results):
                with tabs[i]:
                    # 1. Expander pour le contenu Markdown
                    with st.expander("Contenu de la page", expanded=True):
                        st.markdown(f"{page.url}")
                        st.markdown(page.markdown)

                    # 2. Expander pour les liens
                    with st.expander("Liens trouvés"):
                        if page.links and "internal" in page.links:
                            st.markdown("##### Liens internes")
                            internal_links = page.links["internal"]
                            for link in internal_links:
                                if link["text"]:
                                    st.markdown(
                                        f"- [{link['text']}]({link['href']})"
                                    )

                        if page.links and "external" in page.links:
                            st.markdown("##### Liens externes")
                            external_links = page.links["external"]
                            for link in external_links:
                                text = (
                                    link["text"]
                                    or link["title"]
                                    or link["href"]
                                )
                                st.markdown(f"- [{text}]({link['href']})")

                    # 3. Expander pour les PDFs
                    with st.expander("Fichiers PDF"):
                        if page.media and "images" in page.media:
                            pdf_files = [
                                img
                                for img in page.media["images"]
                                if img.get("format") == "pdf"
                            ]
                            if pdf_files:
                                for pdf in pdf_files:
                                    st.markdown(
                                        f"- [{pdf['desc'] or pdf['src']}]({pdf['src']})"
                                    )
                            else:
                                st.info("Aucun fichier PDF trouvé")
                        else:
                            st.info("Aucun fichier PDF trouvé")

                    # 4. Expander pour les images
                    with st.expander("Images"):
                        if page.media and "images" in page.media:
                            images = [
                                img
                                for img in page.media["images"]
                                if img.get("format") != "pdf"
                            ]
                            unique_images = {}
                            for img in images:
                                group_id = img["group_id"]
                                if group_id not in unique_images or (
                                    img["width"]
                                    and unique_images[group_id]["width"]
                                    and img["width"]
                                    > unique_images[group_id]["width"]
                                ):
                                    unique_images[group_id] = img

                            if unique_images:
                                for img in unique_images.values():
                                    with st.container():
                                        st.image(
                                            img["src"],
                                            caption=img["desc"] or img["alt"],
                                        )
                            else:
                                st.info("Aucune image trouvée")
                        else:
                            st.info("Aucune image trouvée")
        except Exception as e:
            st.error(f"❌ Une erreur s'est produite : {str(e)}")
