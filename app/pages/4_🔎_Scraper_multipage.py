import asyncio

import nest_asyncio
import streamlit as st

# Initialize the event loop before importing crawl4ai
# flake8: noqa: E402
nest_asyncio.apply()

from utils.crawler_utils import CrawlerManager
from utils.dataframe_utils import filter_dataframe
from utils.db_utils import load_urls_data_from_db

# Init crawler
if "crawler_manager" not in st.session_state:

    def reset_crawler_callback():
        st.session_state.crawler_manager = None

    st.session_state.crawler_manager = CrawlerManager(
        on_crawler_reset=reset_crawler_callback
    )
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)


st.title("Scraper html multipage")

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
    help="Sélectionnez les mots-clés qui seront utilisés pour la recherche dans les urls despages web",
)

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
        "type_d_usagers_recherche_d_emplois": "Usagers recherche d'emplois",
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
    url = selected_url["site_web_principal"]
    if not url.startswith(("http://", "https://")):
        st.error("L'URL doit commencer par http:// ou https://")
        st.stop()
    scrape_button = st.button(
        "🔍 Extraire les informations de tarification", use_container_width=True
    )


if scrape_button:
    keywords = st.session_state.selected_keywords

    with st.spinner(
        "Extraction en cours... " "Cela peut prendre quelques instants."
    ):
        try:
            loop = st.session_state.loop
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                st.session_state.crawler_manager.fetch_content(url, keywords)
            )
            # Créer un onglet par page
            tabs = st.tabs([f"Page {i+1}" for i in range(len(results))])

            for i, page in enumerate(results):
                with tabs[i]:
                    # 1. Expander for the markdown content
                    with st.expander("Contenu de la page", expanded=True):
                        st.markdown(f"{page.url}")
                        st.markdown(page.markdown)

                    # 2. Expander for the links
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

                    # 3. Expander for the PDFs
                    with st.expander("Fichiers PDF"):
                        if page.media and "images" in page.media:
                            pdf_files = [
                                img
                                for img in page.media["images"]
                                if img.get("format") == "pdf"
                            ]
                            if pdf_files:
                                for pdf in pdf_files:
                                    print("todo")
                            else:
                                st.info("Aucun fichier PDF trouvé")
                        else:
                            st.info("Aucun fichier PDF trouvé")

                    # 4. Expander for the images
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
