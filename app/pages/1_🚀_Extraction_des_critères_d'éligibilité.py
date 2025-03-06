import asyncio
import base64
import datetime
import io
import os

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from utils.dataframe_utils import filter_dataframe
from utils.grist_utils import get_aoms_data

# from PIL import Image

load_dotenv()

st.title("Extraction des critères d'éligibilité")


def ask_claude_to_extract_table_from_images(images):
    """Demande à Claude d'extraire et structurer les données du tableau."""
    try:
        # Préparer les images
        content = [
            {
                "type": "text",
                "text": (
                    "Tu es un expert en extraction de données tabulaires."
                    "Extrais les données tabulaires de ces images et "
                    "retourne-les uniquement au format CSV, sans autre texte."
                    "Ignore tout ce qui n'est pas une donnée tabulaire."
                ),
            }
        ]
        # Ajouter chaque image au contenu
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64,
                    },
                }
            )
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        # Extraire le contenu du message
        if isinstance(message.content, list):
            for content in message.content:
                if content.type == "text":
                    return content.text
            raise ValueError("Aucun contenu texte trouvé dans la réponse")
        return message.content
    except Exception as e:
        st.error(
            f"Erreur détaillée dans ask_claude_to_extract_table: {str(e)}"
        )
        st.write("Structure de la réponse:", message)
        raise


# Upload des images
# uploaded_files = st.file_uploader(
#     "Choisissez une ou plusieurs images",
#     type=["png", "jpg", "jpeg"],
#     accept_multiple_files=True,
# )
# mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# if uploaded_files:
#     try:
#         # Charger toutes les images
#         images = [Image.open(file) for file in uploaded_files]
#         # Afficher les images individuelles
#         st.subheader("Images téléchargées")
#         cols = st.columns(min(len(images), 3))
#         for idx, (img, col) in enumerate(zip(images, cols), 1):
#             with col:
#                 st.image(img, caption=f"Image {idx}", use_column_width=True)
#         # Extraire les données des tableaux
#         csv_data = ask_claude_to_extract_table_from_images(images)
#         # Convertir en DataFrame
#         df = pd.read_csv(io.StringIO(csv_data))
#         # Afficher le résultat
#         st.success("✅ Données extraites avec succès")
#         st.dataframe(df)
#     except Exception as e:
#         st.error(f"❌ Erreur lors du traitement : {str(e)}")
# else:
#     st.info("👆 Veuillez télécharger une ou plusieurs images")


async def scrape_aom_tarification(url, aom_name):
    """
    Scrape les informations de tarification d'une AOM à partir de son URL.
    Args:
        url: URL de la page de tarification
        aom_name: Nom de l'AOM
    Returns:
        dict: Données structurées extraites au format Markdown
    """
    if not url or pd.isna(url) or url.strip() == "":
        return {"error": "URL non définie pour cette AOM"}
    try:
        async with async_playwright() as p:
            # Lancer le navigateur
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Accéder à la page
            try:
                await page.goto(url, timeout=30000)
            except Exception as e:
                await browser.close()
                return {"error": f"Impossible d'accéder à l'URL: {str(e)}"}

            # Capturer le contenu initial et une capture d'écran
            screenshot = await page.screenshot()

            # Utiliser LLM pour identifier les liens pertinents
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            # Préparation du contenu pour Claude
            content_for_llm = [
                {
                    "type": "text",
                    "text": (
                        f"Analyse cette capture d'écran pour l'AOM {aom_name}."
                        " Identifie les liens qui pourraient contenir des "
                        " informations sur les tarifs, abonnements ou "
                        " conditions d'éligibilité. Retourne uniquement les "
                        " sélecteurs CSS ou XPath de ces liens, un par ligne."
                    ),
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(screenshot).decode("utf-8"),
                    },
                },
            ]

            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": content_for_llm}],
            )

            # Extraire les sélecteurs
            selectors = response.content[0].text.strip().split("\n")

            # Cliquer sur chaque lien identifié et capturer les écrans
            all_screenshots = [
                screenshot
            ]  # Inclure la capture d'écran initiale
            page_titles = ["Page d'accueil"]

            for selector in selectors:
                try:
                    selector = selector.strip()
                    if not selector:
                        continue

                    # Vérifier si le sélecteur existe
                    element_count = await page.evaluate(
                        f"document.querySelectorAll(`{selector}`).length"
                    )

                    if element_count == 0:
                        continue

                    # Obtenir le texte du lien pour le titre
                    link_text = await page.evaluate(
                        f"document.querySelector(`{selector}`).innerText || "
                        f"document.querySelector(`{selector}`).textContent || "
                        f"'Page de tarification'"
                    )

                    # Cliquer sur l'élément
                    await page.click(selector)
                    await page.wait_for_load_state(
                        "networkidle", timeout=10000
                    )

                    # Capturer la page après clic
                    new_screenshot = await page.screenshot(full_page=True)
                    all_screenshots.append(new_screenshot)
                    page_titles.append(link_text.strip())

                    # Revenir à la page précédente
                    await page.go_back()
                    await page.wait_for_load_state(
                        "networkidle", timeout=10000
                    )

                except Exception as e:
                    print(f"Erreur lors du clic sur {selector}: {e}")

            # Analyser les captures d'écran avec LLM
            final_content = [
                {
                    "type": "text",
                    "text": (
                        "Tu es un expert en extraction d'informations"
                        " tarifaires pour les transports publics.\n\n"
                        "Analyse ces captures d'écran pour l'AOM "
                        f"{aom_name}.\n"
                        "Extrait toutes les informations sur:\n"
                        "1. Les tarifs et abonnements\n"
                        "2. Les conditions d'éligibilité "
                        "(âge, revenus, statut)\n"
                        "3. Les justificatifs requis\n\n"
                        "Présente ces informations au format Markdown bien "
                        "structuré avec:\n"
                        "- Des titres et sous-titres clairs\n"
                        "- Des listes à puces pour les éléments\n"
                        "- Des tableaux pour les grilles tarifaires "
                        "si nécessaire\n"
                        "- Des sections distinctes pour chaque type "
                        "d'information\n\n"
                        "Assure-toi que le format est propre et facile à lire."
                    ),
                }
            ]
            # Ajouter les captures d'écran avec leurs titres
            for i, (screenshot, title) in enumerate(
                zip(all_screenshots[:5], page_titles[:5])
            ):
                final_content.append(
                    {
                        "type": "text",
                        "text": f"--- Capture d'écran {i+1}: {title} ---",
                    }
                )
                final_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(screenshot).decode(
                                "utf-8"
                            ),
                        },
                    }
                )
            final_response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0,
                messages=[{"role": "user", "content": final_content}],
            )
            # Extraire le contenu Markdown
            markdown_content = final_response.content[0].text
            # Créer un dictionnaire structuré avec le contenu Markdown
            structured_data = {
                "markdown_content": markdown_content,
                "aom_name": aom_name,
                "source_url": url,
                "extraction_date": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            await browser.close()
            return structured_data
    except Exception as e:
        return {"error": f"Erreur lors du scraping: {str(e)}"}


# Chargement initial des données
aoms_data = get_aoms_data()

# Barre de recherche
search_term = st.text_input(
    "🔍 Rechercher dans toutes les colonnes",
    placeholder="Exemple : Bordeaux Métropole",
)

# Filtrer les données
filtered_df = filter_dataframe(aoms_data, search_term)

# Afficher le nombre de résultats
nb_results = len(filtered_df)
first = f"{nb_results} résultat{'s' if nb_results > 1 else ''}"
second = f"trouvé{'s' if nb_results > 1 else ''}"

st.write(f"📊{first} {second}")


# Afficher le tableau filtré
st.dataframe(
    filtered_df,
    column_config={
        "N_SIREN_AOM": "SIREN AOM",
        "Nom_de_l_AOM": "Nom de l'AOM",
        "Region": "Région",
        "Page_tarification": "URL Tarification",
    },
    hide_index=True,
    use_container_width=True,
)

# Sélection d'une AOM via une liste déroulante
st.subheader("Sélectionner une AOM pour l'extraction")
aom_options = filtered_df["Nom_de_l_AOM"].tolist()
selected_aom_name = st.selectbox("Choisir une AOM", aom_options)


# Récupérer les informations de l'AOM sélectionnée
if selected_aom_name:
    selected_aom = filtered_df[
        filtered_df["Nom_de_l_AOM"] == selected_aom_name
    ].iloc[0]
    st.write(f"### AOM sélectionnée : {selected_aom['Nom_de_l_AOM']}")
    # Vérifier si l'URL est disponible
    url_input_key = f"url_input_{selected_aom_name}"
    if (
        pd.isna(selected_aom.get("Page_tarification"))
        or selected_aom.get("Page_tarification", "").strip() == ""
    ):
        st.warning("Aucune URL de tarification n'est définie pour cette AOM")
        # Demander l'URL à l'utilisateur
        url = st.text_input(
            "Veuillez saisir l'URL du site de transport pour cette AOM:",
            key=url_input_key,
            placeholder="https://www.example.com",
        )
        if not url:
            st.info("Saisissez une URL pour continuer.")
            st.stop()
    else:
        url = selected_aom["Page_tarification"]
        # Permettre à l'utilisateur de modifier l'URL si nécessaire
        url = st.text_input(
            "URL de tarification (modifiable si nécessaire):",
            value=url,
            key=url_input_key,
        )
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
                result = asyncio.run(
                    scrape_aom_tarification(url, selected_aom["Nom_de_l_AOM"])
                )
                # Afficher les résultats
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success("✅ Extraction réussie !")
                    # Afficher le contenu Markdown
                    st.markdown(result["markdown_content"])
                    # Informations sur la source
                    st.caption(
                        f"Source: {result['source_url']} | "
                        f"Extraction effectuée le {result['extraction_date']}"
                    )
                    # Option pour télécharger le contenu Markdown
                    st.download_button(
                        label="📥 Télécharger au format Markdown",
                        data=result["markdown_content"],
                        file_name=f"{result['aom_name']}_tarifs.md",
                        mime="text/markdown",
                    )
            except Exception as e:
                st.error(f"❌ Une erreur s'est produite : {str(e)}")
                st.info("Vérifiez que l'URL est correcte")
