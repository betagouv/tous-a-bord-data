import asyncio
import os
from datetime import datetime
from typing import Dict, List

import nest_asyncio

# Initialize the event loop before importing crawl4ai
# flake8: noqa: E402
nest_asyncio.apply()

import streamlit as st
import tiktoken
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from star_ratings import star_ratings
from streamlit_tags import st_tags

from constants.keywords import DEFAULT_KEYWORDS
from services.evaluation_service import evaluation_service

# Import pour l'évaluation HITL
from services.grist_service import GristDataService
from services.llm_services import LLM_MODELS

# Import pour la classification TSST
from services.tsst_spacy_llm_task import TSSTClassifier
from utils.crawler_utils import CrawlerManager

load_dotenv()


async def get_aom_transport_offers():
    try:
        # Get GristDataService instance
        grist_service = GristDataService.get_instance(
            api_key=os.getenv("GRIST_API_KEY")
        )
        doc_id = os.getenv("GRIST_DOC_INTERMEDIARY_ID")
        aoms = await grist_service.get_aom_transport_offers(doc_id)
        return aoms
    except Exception as e:
        st.error(f"Erreur lors de la connexion à Grist : {str(e)}")
        return []


@traceable
def extract_content(url_source, keywords):
    """
    Extract content from a URL using the crawler.

    Args:
        url_source: The URL to crawl
        keywords: List of keywords to search for

    Returns:
        List of pages with extracted content
    """
    run = get_current_run_tree()
    st.session_state.run_ids["scraping"] = run.id
    try:
        # WARNING: when launch a crawler in streamlit
        # we need a single event loop
        loop = st.session_state.loop
        asyncio.set_event_loop(loop)

        # Réutiliser la même instance de CrawlerManager
        crawler_manager = CrawlerManager()

        pages = loop.run_until_complete(
            crawler_manager.fetch_content(
                url_source,
                keywords,
            )
        )
        return pages
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du contenu : {str(e)}")
        return []


def count_tokens(text: str) -> int:
    """Compte le nombre de tokens dans un texte (version générale)"""
    # Utiliser cl100k comme tokenizer général
    return len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text))


@traceable
def filter_nlp(
    content: str,
) -> Dict[str, str]:
    """Filtre le contenu avec SpaCy"""
    try:
        run = get_current_run_tree()
        st.session_state.run_ids["filter"] = run.id
        from services.nlp_services import (
            extract_markdown_text,
            filter_transport_fare,
            load_spacy_model,
            normalize_text,
        )

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


def check_transport_fare_content(text: str) -> tuple[bool, list]:
    """Vérifie si le contenu contient des informations sur les tarifs de transport

    Returns:
        tuple: (has_fares, fare_matches) où has_fares est un booléen indiquant si des tarifs ont été trouvés
               et fare_matches est une liste des correspondances avec leur contexte
    """
    from services.nlp_services import (
        create_transport_fare_matcher,
        load_spacy_model,
    )

    nlp = load_spacy_model()
    doc = nlp(text)

    # Utiliser le matcher de tarifs de transport
    matcher = create_transport_fare_matcher(nlp)

    # Vérifier s'il y a des correspondances
    matches_regex = matcher(doc)

    # Préparer les informations sur les correspondances pour affichage ultérieur
    fare_matches = []

    for match_id, start, end in matches_regex:
        match_type = nlp.vocab.strings[match_id]
        matched_text = doc[start:end].text

        # Trouver la phrase contenant la correspondance
        sent = None
        for s in doc.sents:
            if start >= s.start and end <= s.end:
                sent = s
                break

        match_info = {
            "type": match_type,
            "text": matched_text,
            "context": None,
        }

        if sent:
            # Préparer le contexte avec le texte mis en évidence
            before = sent.text[: doc[start].idx - sent.start_char]
            match = matched_text
            after = sent.text[
                doc[end - 1].idx + len(doc[end - 1].text) - sent.start_char :
            ]

            match_info["context"] = {
                "before": before,
                "match": match,
                "after": after,
            }

        fare_matches.append(match_info)

    # Si on trouve au moins une correspondance, le contenu contient des informations sur les tarifs
    return len(matches_regex) > 0, fare_matches


@traceable
def format_tags_and_providers(
    text: str, siren: str, name: str
) -> tuple[List[str], List[str]]:
    """Extrait les tags ET les fournisseurs en une seule fois optimisée"""
    run = get_current_run_tree()
    st.session_state.run_ids["format_tags_and_providers"] = run.id
    from services.nlp_services import (
        extract_tags_and_providers,
        load_spacy_model,
    )

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


def show_evaluation_interface(step_name: str) -> None:
    """Affiche l'interface d'évaluation pour une étape"""
    st.subheader("✨ Évaluation")

    # Score avec star_ratings
    stars = star_ratings("Évaluation", numStars=5, key=f"stars_{step_name}")
    quality_score = stars / 5 if stars is not None else 0

    # Commentaire
    correction = st.text_area(
        "Commentaire (optionnel)",
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


# UI
st.set_page_config(
    page_title="Tags extraction - pipeline", layout="wide", page_icon="🏷️"
)
st.header("🏷️ Tags extraction - pipeline")
st.subheader("Sélection de l'AOM")

# init run_ids for evaluation
if "run_ids" not in st.session_state:
    st.session_state.run_ids = {}

# init crawler event loop
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()


aoms = asyncio.run(get_aom_transport_offers())

selected_aom = st.selectbox(
    "Sélectionner une AOM:",
    options=[aom.n_siren_groupement for aom in aoms],
    format_func=lambda x: (
        f"{x} - "
        f"{next((a.nom_aom for a in aoms if a.n_siren_groupement == x), 'Unknown')} "
        f"({sum(1 for a in aoms if a.n_siren_groupement == x and a.site_web_principal)} source"
        f"{'s' if sum(1 for a in aoms if a.n_siren_groupement == x and a.site_web_principal) > 1 else ''})"
    ),
    key="selected_aom",
    on_change=lambda: (
        # Nettoyer toutes les données de session quand on change d'AOM
        st.session_state.pop("raw_scraped_content", None),
        st.session_state.pop("scraped_content", None),
        st.session_state.pop("filtered_contents", None),
        st.session_state.pop(
            "tsst_classification_result", None
        ),  # Réinitialiser la classification TSST
        st.session_state.pop("tags", None),
        st.session_state.pop("providers", None),
        st.session_state.pop(
            "tags_explanations", None
        ),  # Ajout de cette ligne
        st.session_state.pop(
            "providers_explanations", None
        ),  # Ajout de cette ligne
        st.session_state.pop("run_ids", {}),  # Réinitialiser les run_ids
        # Réinitialiser les variables de détection de changement de modèle
        st.session_state.pop("previous_model_name", None),
        st.session_state.pop("model_changed", None),
    ),
)

if selected_aom:
    n_siren_aom = next(
        (a.n_siren_aom for a in aoms if a.n_siren_groupement == selected_aom),
        None,
    )
    nom_aom = next(
        (a.nom_aom for a in aoms if a.n_siren_groupement == selected_aom),
        "Nom inconnu",
    )
    # Collect all site_web_principal values for this AOM
    sources = " | ".join(
        [
            a.site_web_principal
            for a in aoms
            if a.n_siren_groupement == selected_aom and a.site_web_principal
        ]
    )
    st.write("Sources pour cet AOM:")
    for source in sources.split(" | "):
        if source:  # Only display non-empty sources
            st.write(f"- {source}")

    # Step 1: Scraping
    if "available_keywords" not in st.session_state:
        st.session_state.available_keywords = DEFAULT_KEYWORDS.copy()
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = DEFAULT_KEYWORDS.copy()

    with st.expander("🕸️ Task 1 : Scraper le contenu"):
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
        start_button = st.button(
            "🕷️ Lancer l'extraction",
            help="Cliquez pour lancer l'extraction des données sur les sites web",
        )
        if start_button:
            st.session_state.raw_scraped_content = {}
            for url_source in sources.split(" | "):
                st.session_state.raw_scraped_content[url_source] = []
                pages = extract_content(
                    url_source, st.session_state.selected_keywords
                )

                # Ajouter les pages à la liste globale
                for page in pages:
                    st.session_state.raw_scraped_content[url_source].append(
                        {
                            "url": page.url,
                            "markdown": page.markdown,
                        }
                    )
            st.success("✅ Extraction terminée")

    # Step 2: Affichage du contenu scrapé
    with st.expander("👀 Task 2 : Afficher le contenu scrapé"):
        if "raw_scraped_content" in st.session_state:

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

            total_pages = sum(len(pages) for pages in sources.values())

            # Vérifier s'il y a au moins une page avant de créer les onglets
            if total_pages > 0:
                tabs = st.tabs([f"Page {i+1}" for i in range(total_pages)])

                # Compteur pour suivre l'index de l'onglet actuel
                tab_index = 0

                # Parcourir chaque source et ses pages
                for i, (url_source, pages) in enumerate(sources.items()):
                    # Afficher chaque page de la source dans un onglet distinct
                    for page in pages:
                        with tabs[tab_index]:
                            st.write(f"Source {i+1}")
                            st.write(
                                f"Date d'extraction: {datetime.now().strftime('%Y-%m-%d')}"
                            )
                            st.write(f"URL source: {url_source}")
                            st.write(f"URL: {page['url']}")
                            st.markdown(page["markdown"])
                        tab_index += 1
            else:
                st.warning(
                    "⚠️ Aucune page n'a été extraite. Essayez de relancer l'extraction ou de choisir une autre AOM."
                )

            # Sauvegarder dans session_state pour les étapes suivantes
            st.session_state.scraped_content = scraped_content

    # Step 3: Filtrage du contenu
    with st.expander("🎯 Task 3 : Filtrage du contenu"):
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
            has_fares, fare_matches = check_transport_fare_content(
                filtered_content
            )
            # Stocker les résultats dans la session pour affichage après le spinner
            st.session_state.fare_check_result = {
                "has_fares": has_fares,
                "fare_matches": fare_matches,
            }
            nb_tokens = count_tokens(filtered_content)

            st.text_area(
                label=f"Contenu filtré (NLP) - {nb_tokens} tokens",
                value=filtered_content,
                height=500,
                disabled=True,
            )
            if not has_fares:
                st.error(
                    "⚠️ Aucune information sur les tarifs de transport n'a été détectée dans le contenu filtré."
                )
                show_evaluation_interface("filter")
                st.stop()

            show_evaluation_interface("filter")

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
            filtered_result = filter_nlp(scraped_content)
            if filtered_result["Contenu filtré"].strip():
                st.session_state.filtered_contents = filtered_result
                st.success("Filtrage terminé")
                st.rerun()
            else:
                st.error("Aucun contenu pertinent trouvé dans les sources")

    with st.expander("🤖 Task 4 : Classification TSST avec LLM"):
        # Vérifier si l'étape précédente est complétée
        is_previous_step_complete = (
            "filtered_contents" in st.session_state
            and st.session_state.filtered_contents.get(
                "Contenu filtré", ""
            ).strip()
        )

        if not is_previous_step_complete:
            st.warning("⚠️ Veuillez d'abord compléter l'étape de filtrage")

        # Fonction pour détecter le changement de modèle
        def on_model_change():
            if (
                "previous_model_name" in st.session_state
                and st.session_state.selected_model_name
                != st.session_state.previous_model_name
            ):
                # Réinitialiser le résultat de classification si le modèle change
                st.session_state.pop("tsst_classification_result", None)
                st.session_state.model_changed = (
                    True  # Set this to True when model changes
                )
                st.session_state.previous_model_name = (
                    st.session_state.selected_model_name
                )

        # Initialiser les variables de session si elles n'existent pas
        if "previous_model_name" not in st.session_state:
            st.session_state.previous_model_name = list(LLM_MODELS.keys())[0]
        if "model_changed" not in st.session_state:
            st.session_state.model_changed = False

        # Sélecteur de modèle LLM avec détection de changement
        selected_model_name = st.selectbox(
            "Modèle LLM à utiliser",
            options=list(LLM_MODELS.keys()),
            index=0,
            key="selected_model_name",
            on_change=on_model_change,
        )

        # Afficher le résultat de la classification TSST s'il existe
        if "tsst_classification_result" in st.session_state:
            result = st.session_state.tsst_classification_result

            st.subheader("Résultat de la classification TSST")

            if result["is_tsst"]:
                st.success(
                    "✅ Le contenu concerne la tarification sociale et solidaire des transports (TSST)"
                )
            else:
                st.error(
                    "❌ Le contenu ne concerne PAS la tarification sociale et solidaire des transports"
                )
                show_evaluation_interface("tsst_classification")
                st.stop()

            # Afficher la justification si disponible
            if "justification" in result and result["justification"]:
                st.markdown("**Justification:**")
                st.info(result["justification"])

            # Créer un conteneur pour les détails techniques
            st.markdown("**Détails techniques:**")
            col1, col2 = st.columns(2)

            st.markdown("**Réponse du LLM:**")
            st.code(result["response"], language="text")

            # Ajouter l'interface d'évaluation
            show_evaluation_interface("tsst_classification")

        @traceable
        def classify_tsst(content: str, model_name: str) -> Dict:
            """Classifie le contenu pour déterminer s'il concerne la TSST"""
            run = get_current_run_tree()
            st.session_state.run_ids["tsst_classification"] = run.id

            # Initialiser le classifieur TSST
            classifier = TSSTClassifier(model_name=model_name)

            # Classifier le contenu entier
            is_tsst, details = classifier.classify_paragraph(content)

            # Préparer le résultat
            result = {
                "is_tsst": is_tsst,
                "scores": details["scores"],
                "prompt": details["prompt"],
                "response": details["response"],
                "justification": details.get("justification", ""),
            }

            return result

        if st.button(
            "Lancer la classification TSST",
            key="tsst_classification",
            disabled=not is_previous_step_complete,
        ):
            # Vérification du contenu filtré
            filtered_content = st.session_state.filtered_contents.get(
                "Contenu filtré", ""
            ).strip()
            if not filtered_content:
                st.error("Le contenu filtré est vide")
                st.stop()
            with st.spinner("Classification TSST en cours..."):
                try:
                    # Appeler la fonction traçable
                    result = classify_tsst(
                        filtered_content, selected_model_name
                    )
                    # Stocker le résultat dans la session
                    st.session_state.tsst_classification_result = result
                    st.success("Classification TSST terminée")
                    st.rerun()
                except Exception as e:
                    st.error(
                        f"Erreur lors de la classification TSST: {str(e)}"
                    )

    with st.expander(
        "🏷️ Task 5 : Extraction des tags et fournisseurs", expanded=True
    ):
        # Vérifier si la classification TSST est activée et disponible
        tsst_enabled = "tsst_classification_result" in st.session_state

        # Vérifier si l'étape précédente est complétée
        is_previous_step_complete = (
            "filtered_contents" in st.session_state
            and st.session_state.filtered_contents.get(
                "Contenu filtré", ""
            ).strip()
        )

        # Si la classification TSST est activée mais le résultat est négatif, bloquer l'extraction
        if (
            tsst_enabled
            and not st.session_state["tsst_classification_result"]["is_tsst"]
        ):
            st.error(
                "❌ Le contenu ne concerne pas la tarification sociale et solidaire des transports. L'extraction des tags est désactivée."
            )
            is_previous_step_complete = False
            st.stop()
        elif not is_previous_step_complete:
            st.warning("⚠️ Veuillez d'abord compléter l'étape de filtrage")

        # Créer un conteneur pour le bouton d'extraction
        extraction_container = st.container()
        with extraction_container:
            if st.button(
                "Extraire les tags et fournisseurs",
                key="format_tags_and_providers",
                use_container_width=True,
                disabled=not is_previous_step_complete,
            ):
                with st.spinner("Extraction en cours..."):
                    # Vérifier d'abord si le contenu contient des informations sur les tarifs
                    filtered_content = st.session_state.filtered_contents[
                        "Contenu filtré"
                    ].strip()

                    # UNE SEULE extraction pour tout
                    tags, providers = format_tags_and_providers(
                        filtered_content,
                        n_siren_aom,
                        nom_aom,
                    )
                    st.session_state.tags = tags
                    st.session_state.providers = providers
                    st.rerun()

                # Afficher les résultats de la vérification des tarifs (ce code ne sera exécuté qu'après le spinner)
                if "fare_check_result" in st.session_state:
                    result = st.session_state.fare_check_result

                    st.subheader(
                        "Détails des correspondances de tarifs détectées"
                    )

                    if result["has_fares"]:
                        st.success(
                            f"✅ {len(result['fare_matches'])} correspondances de tarifs trouvées"
                        )

                        for match in result["fare_matches"]:
                            st.markdown(f"**Type de tarif**: {match['type']}")
                            st.markdown(f"**Texte détecté**: {match['text']}")

                            if match["context"]:
                                context = match["context"]
                                st.markdown(
                                    f"**Contexte**: {context['before']}<mark style='background-color: #FFFF00'>{context['match']}</mark>{context['after']}",
                                    unsafe_allow_html=True,
                                )

                            st.markdown("---")
                    else:
                        st.warning(
                            "⚠️ Aucune correspondance de tarif trouvée dans le texte"
                        )

        # Créer des onglets pour organiser le contenu
        if (
            "tags" in st.session_state
            or "providers" in st.session_state
            or "fare_check_result" in st.session_state
        ):
            tabs = st.tabs(
                [
                    "Tags détectés",
                    "Fournisseurs détectés",
                    "Tarifs détectés",
                    "Évaluation",
                ]
            )

            # Onglet des tags
            with tabs[0]:
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
                        st.markdown("#### ℹ️ Explications des tags détectés")
                        # Créer un conteneur HTML scrollable
                        explanation_html = "<div class='scrollable-container'>"
                        for (
                            tag,
                            match_info,
                        ) in st.session_state.tags_explanations[
                            "matches"
                        ].items():
                            explanation_html += (
                                f"<p><strong>{tag}</strong> détecté dans :</p>"
                            )
                            explanation_html += f"{match_info}<hr>"
                        explanation_html += "</div>"
                        # Utiliser components.v1.html au lieu de markdown
                        st.components.v1.html(
                            explanation_html, height=400, scrolling=True
                        )

            # Onglet des fournisseurs
            with tabs[1]:
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
                        st.markdown(
                            "#### ℹ️ Explications des fournisseurs détectés"
                        )
                        # Créer un conteneur HTML scrollable
                        explanation_html = "<div class='scrollable-container'>"
                        for (
                            provider,
                            match_info,
                        ) in st.session_state.providers_explanations[
                            "matches"
                        ].items():
                            explanation_html += f"<p><strong>{provider}</strong> détecté dans :</p>"
                            explanation_html += f"{match_info}<hr>"
                        explanation_html += "</div>"
                        # Utiliser components.v1.html au lieu de markdown
                        st.components.v1.html(
                            explanation_html, height=400, scrolling=True
                        )

            # Onglet des tarifs détectés
            with tabs[2]:
                if "fare_check_result" in st.session_state:
                    result = st.session_state.fare_check_result

                    st.markdown("### 💰 Tarifs détectés")

                    if result["has_fares"]:
                        st.success(
                            f"✅ {len(result['fare_matches'])} correspondances de tarifs trouvées"
                        )

                        for match in result["fare_matches"]:
                            st.markdown(f"**Type de tarif**: {match['type']}")
                            st.markdown(f"**Texte détecté**: {match['text']}")

                            if match["context"]:
                                context = match["context"]
                                st.markdown(
                                    f"**Contexte**: {context['before']}<mark style='background-color: #FFFF00'>{context['match']}</mark>{context['after']}",
                                    unsafe_allow_html=True,
                                )

                            st.markdown("---")
                    else:
                        st.warning(
                            "⚠️ Aucune correspondance de tarif trouvée dans le texte"
                        )

            # Onglet d'évaluation
            with tabs[3]:
                if (
                    "tags" in st.session_state
                    and "providers" in st.session_state
                ):
                    show_evaluation_interface("format_tags_and_providers")
