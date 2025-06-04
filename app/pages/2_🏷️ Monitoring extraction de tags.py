import asyncio
import os
import re
from datetime import datetime
from typing import Dict, List

import nest_asyncio
import streamlit as st
from streamlit_tags import st_tags

# Initialize the event loop before importing crawl4ai
# flake8: noqa: E402
nest_asyncio.apply()

import tiktoken
from anthropic import Anthropic
from constants.entites_eligibilite import ENTITES
from constants.keywords import DEFAULT_KEYWORDS
from constants.tag_dp_mapping import TAG_DP_MAPPING
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# Import pour l'évaluation HITL
from services.evaluation_service import evaluation_service

# Nouveaux imports
from services.llm_services import (
    LLM_MODELS,
    MAX_TOKEN_OUTPUT,
    call_anthropic,
    call_ollama,
    call_scaleway,
)
from services.nlp_services import (
    extract_markdown_text,
    extract_tags_and_providers,
    filter_transport_fare,
    load_spacy_model,
    normalize_text,
)
from sqlalchemy import create_engine, text
from star_ratings import star_ratings
from utils.crawler_utils import CrawlerManager
from utils.db_utils import get_postgres_cs
from utils.streamlit_utils import scroll_to_bottom

# Configuration de la page pour utiliser plus de largeur
st.set_page_config(page_title="Extraction des tags", layout="wide")

load_dotenv()


st.title("Extraction des tags")

# Connect to the database
engine = create_engine(get_postgres_cs())


def get_aom_content_by_source(siren: str, source_url: str) -> str:
    """Récupère le contenu d'une source spécifique pour un AOM"""
    with engine.connect() as conn:
        pages = conn.execute(
            text(
                """
                SELECT url_page, contenu_scrape
                FROM tarification_raw
                WHERE n_siren_aom = :siren
                AND url_source = :url
                ORDER BY id
            """
            ),
            {"siren": siren, "url": source_url},
        ).fetchall()
        all_pages = []
        for page in pages:
            all_pages.append(
                f"--- Page: {page.url_page} ---\n{page.contenu_scrape}"
            )
    return "\n\n".join(all_pages)


def count_tokens(text: str) -> int:
    """Compte le nombre de tokens dans un texte (version générale)"""
    # Utiliser cl100k comme tokenizer général
    return len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text))


# Initialiser le dictionnaire des run_ids s'il n'existe pas
if "run_ids" not in st.session_state:
    st.session_state.run_ids = {}


@traceable
def filter_nlp(
    content: str, model: str, siren: str, name: str
) -> Dict[str, str]:
    """Filtre le contenu avec SpaCy"""
    try:
        run = get_current_run_tree()
        st.session_state.run_ids["filter"] = run.id

        with st.spinner("Analyse automatique du langage naturel..."):
            nlp = load_spacy_model()
            raw_text = extract_markdown_text(content)
            paragraphs = normalize_text(raw_text, nlp)
            paragraphs_filtered, _ = filter_transport_fare(paragraphs, nlp)
            filtered = "\n\n".join(paragraphs_filtered)

            if filtered:
                return {"Contenu filtré": filtered}
            else:
                return {"Contenu filtré": ""}
    except Exception as e:
        st.error(f"Erreur de filtrage NLP : {str(e)}")
        return {"Contenu filtré": f"Erreur lors du filtrage NLP : {str(e)}"}


@traceable
def format_tags_and_providers(
    text: str, siren: str, name: str
) -> tuple[List[str], List[str]]:
    """Extrait les tags ET les fournisseurs en une seule passe optimisée"""
    run = get_current_run_tree()
    st.session_state.run_ids["format_tags_and_providers"] = run.id

    nlp = load_spacy_model()

    # UNE SEULE extraction pour tout
    tags, providers, tags_debug, providers_debug = extract_tags_and_providers(
        text, nlp, siren, name
    )

    # Stocker les explications dans session_state
    if tags_debug:
        st.session_state.tags_explanations = {
            "title": "### ℹ️ Explications des tags détectés",
            "matches": {
                tag: match_info
                for tag, match_info in tags_debug.items()
                if tag is not None
            },
        }

    if providers_debug:
        st.session_state.providers_explanations = {
            "title": "### ℹ️ Explications des fournisseurs détectés",
            "matches": {
                provider: match_info
                for provider, match_info in providers_debug.items()
                if provider is not None
            },
        }

    return tags, providers


def show_evaluation_interface(step_name: str, content: str) -> None:
    """Affiche l'interface d'évaluation pour une étape"""
    st.subheader("✨ Évaluation")

    # Score avec star_ratings
    stars = star_ratings("", numStars=5, key=f"stars_{step_name}")
    quality_score = stars / 5 if stars is not None else 0

    # Correction proposée
    correction = st.text_area(
        "Correction proposée (optionnel)",
        placeholder="Proposez une version corrigée du résultat...",
        key=f"correction_{step_name}",
    )

    # Bouton de sauvegarde
    if st.button("💾 Sauvegarder l'évaluation", key=f"save_{step_name}"):
        with st.spinner("Sauvegarde de l'évaluation..."):
            run_id = st.session_state.run_ids.get(step_name)
            if not run_id:
                st.error(f"❌ Pas de run_id trouvé pour l'étape {step_name}")
                return

            feedback = evaluation_service.create_feedback(
                run_id=run_id,
                key="quality",
                score=quality_score,
                correction=correction,
            )
            if feedback:
                st.success("✅ Évaluation sauvegardée !")
            else:
                st.error("❌ Erreur lors de la sauvegarde")


def toggle_crawling():
    if "is_crawling" not in st.session_state:
        st.session_state.is_crawling = False

    st.session_state.is_crawling = not st.session_state.is_crawling
    return st.session_state.is_crawling


# init crawler
if "crawler_manager" not in st.session_state:

    def reset_crawler_callback():
        st.session_state.crawler_manager = None

    st.session_state.crawler_manager = CrawlerManager(
        on_crawler_reset=reset_crawler_callback
    )
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)


# Interface Streamlit
st.subheader("Sélection de l'AOM")

# Get unique AOMs with their URLs
with engine.connect() as conn:
    aoms = conn.execute(
        text(
            """
            SELECT DISTINCT
                t.n_siren_aom,
                a.nom_aom,
                COUNT(DISTINCT t.url_source) as nb_sources,
                STRING_AGG(DISTINCT t.url_source, ' | ') as sources
            FROM tarification_raw t
            LEFT JOIN aoms a ON t.n_siren_aom = a.n_siren_aom
            GROUP BY t.n_siren_aom, a.nom_aom
            ORDER BY COUNT(DISTINCT t.url_source) DESC, a.nom_aom
            """
        )
    ).fetchall()

selected_aom = st.selectbox(
    "Sélectionner une AOM:",
    options=[aom[0] for aom in aoms],
    format_func=lambda x: (
        f"{x} - "
        f"{next((a[1] for a in aoms if a[0] == x), 'Unknown')} "
        f"({next((a[2] for a in aoms if a[0] == x), 0)} sources)"
    ),
    key="selected_aom",
    on_change=lambda: (
        # Nettoyer toutes les données de session quand on change d'AOM
        st.session_state.pop("raw_scraped_content", None),
        st.session_state.pop("scraped_content", None),
        st.session_state.pop("filtered_contents", None),
        st.session_state.pop("tags", None),
        st.session_state.pop("providers", None),
        st.session_state.pop("run_ids", {}),  # Réinitialiser les run_ids
    ),
)

if selected_aom:
    n_siren_aom = next((a[0] for a in aoms if a[0] == selected_aom), None)
    nom_aom = next((a[1] for a in aoms if a[0] == selected_aom), "Nom inconnu")
    sources = next((a[3] for a in aoms if a[0] == selected_aom), "")
    st.write("Sources pour cet AOM:")
    for source in sources.split(" | "):
        st.write(f"- {source}")

    # Step 1: Scraping
    if "available_keywords" not in st.session_state:
        st.session_state.available_keywords = DEFAULT_KEYWORDS.copy()
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = DEFAULT_KEYWORDS.copy()

    with st.expander("🕸️ Étape 1 : Scraper le contenu"):
        new_keyword = st.text_input(
            "Ajouter un nouveau mot-clé :",
            placeholder="Entrez un nouveau mot-clé et appuyez sur Entrée",
            help="Le nouveau mot-clé sera ajouté à la liste disponible",
        )

        if new_keyword:
            if new_keyword not in st.session_state.available_keywords:
                st.session_state.available_keywords.append(new_keyword)
                st.session_state.selected_keywords.append(new_keyword)
                st.rerun()

        selected_keywords = st.multiselect(
            "Mots-clés :",
            options=st.session_state.available_keywords,
            default=st.session_state.selected_keywords,
        )

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
            st.session_state.raw_scraped_content = {}
            for url_source in sources.split(" | "):
                st.session_state.raw_scraped_content[url_source] = []
                st.write(f"URL: {url_source}")
                try:
                    # Scraper l'URL
                    loop = st.session_state.loop
                    asyncio.set_event_loop(loop)
                    pages = loop.run_until_complete(
                        st.session_state.crawler_manager.fetch_content(
                            url_source,
                            st.session_state.selected_keywords,
                        )
                    )
                    # Ajouter les pages à la liste globale
                    for page in pages:
                        st.session_state.raw_scraped_content[
                            url_source
                        ].append(
                            {
                                "url": page.url,
                                "markdown": page.markdown,
                            }
                        )
                except Exception as e:
                    st.error(f"⚠️ Une erreur est survenue : {str(e)}")
            st.session_state.is_crawling = False
            st.rerun()

    # Step 2: Affichage du contenu scrapé
    with st.expander("👀 Étape 2 : Afficher le contenu scrapé"):
        if (
            "raw_scraped_content" in st.session_state
            and st.session_state.raw_scraped_content
        ):
            # Utiliser les données en session
            sources = st.session_state.raw_scraped_content

            # Préparer le contenu total pour compter les tokens
            scraped_content = ""
            for pages in sources.values():
                for page in pages:
                    scraped_content += (
                        f"--- Page: {page['url']} ---\n{page['markdown']}\n\n"
                    )

            nb_tokens = count_tokens(scraped_content)
            st.write(f"Nombre de tokens : {nb_tokens}")

            # Afficher le contenu par source
            tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])

            for i, (url_source, pages) in enumerate(sources.items()):
                with tabs[i]:
                    st.write(f"Source {i+1}")
                    st.write(
                        f"Date d'extraction: {datetime.now().strftime('%Y-%m-%d')}"
                    )
                    st.write(f"URL source: {url_source}")
                    # Afficher chaque page de la source
                    for page in pages:
                        st.write(f"URL: {page['url']}")
                        st.markdown(page["markdown"])

            # Sauvegarder dans session_state pour les étapes suivantes
            st.session_state.scraped_content = scraped_content

        else:
            # Fallback sur la base de données (code existant)
            sources = next(
                (a[3] for a in aoms if a[0] == selected_aom), ""
            ).split(" | ")
            scraped_content = ""
            for i, source in enumerate(sources):
                content = get_aom_content_by_source(selected_aom, source)
                scraped_content += content + "\n\n"
            nb_tokens = count_tokens(scraped_content)
            st.write(f"Nombre de tokens : {nb_tokens}")
            sources_content = {}
            tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])

            # Concaténer le contenu de toutes les sources
            for i, source in enumerate(sources):
                with tabs[i]:
                    st.write(f"URL: {source}")
                    content = get_aom_content_by_source(selected_aom, source)
                    sources_content[source] = content
                    st.markdown(content)

            # Sauvegarder dans session_state pour les étapes suivantes
            st.session_state.scraped_content = scraped_content

    # Step 3: Filtrage du contenu
    with st.expander("🎯 Étape 3 : Filtrage du contenu"):
        # Vérifier si l'étape précédente est complétée
        is_previous_step_complete = (
            "scraped_content" in st.session_state
            and st.session_state.scraped_content
        )

        if not is_previous_step_complete:
            st.warning("⚠️ Veuillez d'abord compléter l'étape de scraping")

        # Afficher le contenu filtré s'il existe
        if "filtered_contents" in st.session_state:
            filtered_content = st.session_state.filtered_contents[
                "Contenu filtré"
            ]
            nb_tokens = count_tokens(filtered_content)

            st.text_area(
                label=f"Contenu filtré (NLP) - {nb_tokens} tokens",
                value=filtered_content,
                height=500,
                disabled=True,
            )
            # Ajouter l'interface d'évaluation
            show_evaluation_interface("filter", filtered_content)

        if st.button(
            "Lancer le filtrage",
            key="filter_content",
            disabled=not is_previous_step_complete,
        ):
            # Vérification du contenu une seule fois
            scraped_content = st.session_state.get("scraped_content", {})
            if not scraped_content:
                st.error("Veuillez d'abord charger le contenu dans l'étape 2")
                st.stop()
            filtered_result = filter_nlp(
                scraped_content,
                "custom_filter_v2",
                n_siren_aom,
                nom_aom,
            )
            if filtered_result["Contenu filtré"].strip():
                st.session_state.filtered_contents = filtered_result
                st.success("Filtrage terminé")
                st.rerun()
            else:
                st.error("Aucun contenu pertinent trouvé dans les sources")

    # Step 4: Extraction des tags ET fournisseurs (combinée)
    with st.expander(
        "🏷️ Étape 4 : Extraction des tags et fournisseurs", expanded=True
    ):
        is_previous_step_complete = (
            "filtered_contents" in st.session_state
            and st.session_state.filtered_contents.get(
                "Contenu filtré", ""
            ).strip()
        )

        if not is_previous_step_complete:
            st.warning("⚠️ Veuillez d'abord compléter l'étape de filtrage")

        if st.button(
            "Extraire les tags et fournisseurs",
            key="format_tags_and_providers",
            use_container_width=True,
            disabled=not is_previous_step_complete,
        ):
            with st.spinner("Extraction en cours..."):
                # UNE SEULE extraction pour tout
                tags, providers = format_tags_and_providers(
                    st.session_state.filtered_contents[
                        "Contenu filtré"
                    ].strip(),
                    n_siren_aom,
                    nom_aom,
                )
                st.session_state.tags = tags
                st.session_state.providers = providers
                st.rerun()

        # Affichage des tags
        if "tags" in st.session_state:
            st.markdown("### 🏷️ Tags détectés")
            st.session_state.tags = st_tags(
                label="",
                text="",
                value=st.session_state.tags,
                key="tag_display",
            )

            # Explications des tags
            if "tags_explanations" in st.session_state:
                for tag, match_info in st.session_state.tags_explanations[
                    "matches"
                ].items():
                    st.markdown(f"**{tag}** détecté dans :")
                    st.markdown(match_info, unsafe_allow_html=True)
                    st.markdown("---")
                scroll_to_bottom()

        # Affichage des fournisseurs
        if "providers" in st.session_state:
            st.markdown("### 📊 Fournisseurs détectés")
            st.session_state.providers = st_tags(
                label="",
                text="",
                value=st.session_state.providers,
                key="provider_display",
            )

            # Explications des fournisseurs
            if "providers_explanations" in st.session_state:
                for (
                    provider,
                    match_info,
                ) in st.session_state.providers_explanations[
                    "matches"
                ].items():
                    st.markdown(f"**{provider}** détecté dans :")
                    st.markdown(match_info, unsafe_allow_html=True)
                    st.markdown("---")
                scroll_to_bottom()

        # Interface d'évaluation combinée
        if "tags" in st.session_state and "providers" in st.session_state:
            show_evaluation_interface(
                "format_tags_and_providers",
                f"Tags: {st.session_state.tags}, Providers: {st.session_state.providers}",
            )
