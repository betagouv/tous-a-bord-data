import asyncio

import nest_asyncio
import streamlit as st

# flake8: noqa: E402
asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CacheMode, CrawlerRunConfig
from utils.dataframe_utils import filter_dataframe
from utils.db_utils import load_urls_data_from_db


async def scraper_multipage(url):

    browser_config = BrowserConfig(verbose=True)

    run_config = CrawlerRunConfig(
        # Content filtering
        word_count_threshold=10,
        exclude_external_links=True,
        # Content processing
        remove_overlay_elements=True,  # Remove popups/modals
        process_iframes=True,  # Process iframe content
        # Cache control
        cache_mode=CacheMode.ENABLED,  # Use cache if available
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
    # Vérifier si l'URL est disponible
    url = selected_url["site_web_principal"]
    # Vérifier que l'URL est valide
    if not url.startswith(("http://", "https://")):
        st.error("L'URL doit commencer par http:// ou https://")
        st.stop()
    # Bouton pour lancer le scraping
    col1, col2 = st.columns([1, 3])
    with col1:
        scrape_button = st.button(
            "🔍 Extraire les critères", use_container_width=True
        )
    if scrape_button:
        with st.spinner(
            "Extraction en cours... " "Cela peut prendre quelques instants."
        ):
            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(scraper_multipage(url))
                if not result.success:
                    st.error(
                        f"❌ Echec de l'extraction: {result.error_message}"
                    )
                with st.expander("Résultats de l'extraction", expanded=True):
                    tab1, tab2, tab3, tab4 = st.tabs(
                        ["Page 1", "Liens internes", "Fichiers pdf", "Images"]
                    )
                    with tab1:
                        st.markdown(result.markdown)
                    with tab2:
                        if result.links and "internal" in result.links:
                            st.markdown("##### Liens internes")
                            internal_links = result.links["internal"]
                            for link in internal_links:
                                if link["text"]:
                                    st.markdown(
                                        f"- [{link['text']}]({link['href']})"
                                    )
                    with tab3:
                        # Section PDFs
                        st.markdown("---")  # Séparateur
                        st.subheader("Fichiers PDF")
                        if result.media and "images" in result.media:
                            pdf_files = [
                                img
                                for img in result.media["images"]
                                if img.get("format") == "pdf"
                            ]
                            if pdf_files:
                                for pdf in pdf_files:
                                    st.markdown(f"- {pdf['desc']}")
                            else:
                                st.info("Aucun fichier PDF trouvé")
                        else:
                            st.info("Aucun fichier PDF trouvé")
                    with tab4:
                        # Section Images
                        st.subheader("Images")
                        if result.media and "images" in result.media:
                            images = result.media["images"]
                            # Filtrer les images uniques par group_id
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
                st.info("Vérifiez que l'URL est correcte")
