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
from constants.keywords import DEFAULT_KEYWORDS
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from services.evaluation_service import evaluation_service

# Import pour l'évaluation HITL
from services.grist_service import GristDataService
from services.llm_services import LLM_MODELS

# Import pour la classification TSST
from services.tsst_spacy_llm_task import TSSTClassifier
from star_ratings import star_ratings
from streamlit_tags import st_tags
from utils.crawler_utils import CrawlerManager

# Configuration de la page pour utiliser plus de largeur
st.set_page_config(page_title="Extraction des tags", layout="wide")
st.title("Extraction des tags")
load_dotenv()


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
        # Run the crawler and get the results with a timeout
        crawler_manager = CrawlerManager()
        # Use asyncio.run() to create a new event loop for each request
        pages = asyncio.run(
            crawler_manager.fetch_content(url_source, keywords)
        )
        return pages
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du contenu : {str(e)}")
        return []


def count_tokens(text: str) -> int:
    """Compte le nombre de tokens dans un texte (version générale)"""
    # Utiliser cl100k comme tokenizer général
    return len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text))


# Initialiser le dictionnaire des run_ids s'il n'existe pas
if "run_ids" not in st.session_state:
    st.session_state.run_ids = {}


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


# Interface Streamlit
st.subheader("Sélection de l'AOM")


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
        os._exit(0),
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

# Section de traitement batch
st.markdown("---")
st.header("🔄 Traitement batch")
st.markdown(
    "Cette section permet de lancer un traitement batch pour plusieurs AOMs en utilisant les configurations définies ci-dessus."
)

# Initialiser les variables de session pour le batch
if "batch_processing_active" not in st.session_state:
    st.session_state.batch_processing_active = False
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

# Récupérer les configurations des expanders
batch_config = {
    "keywords": st.session_state.get(
        "selected_keywords", DEFAULT_KEYWORDS.copy()
    ),
    "model_name": st.session_state.get(
        "selected_model_name", list(LLM_MODELS.keys())[0]
    ),
}

# Afficher les configurations actuelles
st.subheader("Configuration actuelle")
col1, col2 = st.columns(2)
with col1:
    st.write("**Keywords pour scraping:**")
    st.write(", ".join(batch_config["keywords"]))
with col2:
    st.write("**Modèle pour classification:**")
    st.write(batch_config["model_name"])

# Bouton pour lancer le traitement batch
if st.button(
    "🚀 Lancer le traitement batch", type="primary", use_container_width=True
):
    # Réinitialiser les résultats précédents
    st.session_state.batch_results = []
    st.session_state.batch_processing_active = True

    # Conteneurs pour l'affichage en temps réel
    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Fonction pour mettre à jour la progression
        def update_progress(current, total, result):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(
                f"Progression: {current}/{total} AOMs traités ({progress:.1%})"
            )

        # Initialiser le BatchProcessor
        from services.batch_tag_extraction import BatchProcessor

        batch_processor = BatchProcessor(max_workers=4)

        # Lancer le traitement batch
        with st.spinner("Traitement batch en cours..."):
            try:
                # Configurer le batch processor avec les paramètres des expanders
                batch_processor.keywords = batch_config["keywords"]
                batch_processor.model_name = batch_config["model_name"]

                # Lancer le traitement
                results = batch_processor.process_batch(
                    aom_list=aoms[6:10], progress_callback=update_progress
                )

                # Sauvegarder les résultats dans la session
                st.session_state.batch_results = results

                # Convertir les résultats en AomWithTags pour Grist
                aoms_with_tags = []
                for result in results:
                    if result.status == "success":
                        from models.grist_models import AomWithTags

                        # Créer un objet AomWithTags
                        aom_with_tags = AomWithTags(
                            n_siren_groupement=int(result.n_siren_aom),
                            n_siren_aom=int(result.n_siren_aom),
                            nom_aom=result.nom_aom,
                            commune_principale_aom="",  # À compléter si disponible
                            nombre_commune_aom=0,  # À compléter si disponible
                            labels=result.tags,
                            fournisseurs=result.providers,
                            status=result.status,
                        )
                        aoms_with_tags.append(aom_with_tags)

                # Sauvegarder les résultats dans Grist
                if aoms_with_tags:
                    import asyncio
                    import os

                    async def save_to_grist():
                        try:
                            # Get GristDataService instance
                            grist_service = GristDataService.get_instance(
                                api_key=os.getenv("GRIST_API_KEY")
                            )
                            doc_id = os.getenv("GRIST_DOC_OUTPUT_ID")

                            # Mettre à jour les AOMs avec tags dans Grist
                            await grist_service.update_aom_with_tags_batch(
                                aoms_with_tags, doc_id
                            )
                            return True
                        except Exception as e:
                            st.error(
                                f"Erreur lors de la sauvegarde dans Grist: {str(e)}"
                            )
                            return False

                    # Exécuter la fonction asynchrone
                    success = asyncio.run(save_to_grist())
                    if success:
                        st.success("✅ Résultats sauvegardés dans Grist")

                st.success(
                    f"✅ Traitement batch terminé pour {len(results)} AOMs"
                )

            except Exception as e:
                st.error(f"❌ Erreur lors du traitement batch: {str(e)}")
            finally:
                st.session_state.batch_processing_active = False

# Afficher les résultats du traitement batch
if st.session_state.batch_results:
    st.subheader("Résultats du traitement batch")

    # Créer un DataFrame pour l'affichage
    import pandas as pd

    results_data = []
    for result in st.session_state.batch_results:
        # Emoji pour le statut
        status_emoji = {"success": "✅", "error": "❌", "no_data": "⚠️"}.get(
            result.status, "❔"
        )

        results_data.append(
            {
                "SIREN": result.n_siren_aom,
                "Nom AOM": result.nom_aom,
                "Statut": f"{status_emoji} {result.status}",
                "Labels": ", ".join(result.tags) if result.tags else "",
                "Fournisseurs": ", ".join(result.providers)
                if result.providers
                else "",
                "Nb Labels": len(result.tags) if result.tags else 0,
                "Nb Fournisseurs": len(result.providers)
                if result.providers
                else 0,
                "Temps (s)": f"{result.processing_time:.1f}"
                if result.processing_time
                else "",
                "Erreur": result.error_message or "",
            }
        )

    results_df = pd.DataFrame(results_data)

    # Filtres pour le tableau
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filtrer par status",
            ["Tous"]
            + [
                s.replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", "")
                for s in results_df["Statut"].unique()
            ],
            key="batch_status_filter",
        )

    with col2:
        min_tags = st.number_input(
            "Nombre minimum de labels",
            min_value=0,
            value=0,
            key="batch_min_tags_filter",
        )

    with col3:
        search_term = st.text_input(
            "Rechercher dans le nom", key="batch_search_filter"
        )

    # Appliquer les filtres
    filtered_df = results_df.copy()

    if status_filter != "Tous":
        filtered_df = filtered_df[
            filtered_df["Statut"].str.contains(status_filter)
        ]

    if min_tags > 0:
        filtered_df = filtered_df[filtered_df["Nb Labels"] >= min_tags]

    if search_term:
        filtered_df = filtered_df[
            filtered_df["Nom AOM"].str.contains(
                search_term, case=False, na=False
            )
        ]

    # Afficher le tableau filtré
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Bouton pour relancer les AOMs en erreur
    if st.button("🔄 Relancer les AOMs en erreur", key="retry_batch_errors"):
        error_aoms = [
            result.n_siren_aom
            for result in st.session_state.batch_results
            if result.status == "error"
        ]

        if error_aoms:
            st.session_state.batch_processing_active = True

            # Initialiser le BatchProcessor
            from services.batch_tag_extraction import BatchProcessor

            batch_processor = BatchProcessor(max_workers=4)

            # Configurer le batch processor avec les paramètres des expanders
            batch_processor.keywords = batch_config["keywords"]
            batch_processor.model_name = batch_config["model_name"]

            # Conteneur pour la progression
            progress_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Fonction pour mettre à jour la progression
                def update_progress(current, total, result):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Progression: {current}/{total} AOMs traités ({progress:.1%})"
                    )

                # Lancer le traitement pour les AOMs en erreur
                with st.spinner(
                    f"Relancement de {len(error_aoms)} AOMs en erreur..."
                ):
                    try:
                        results = batch_processor.process_batch(
                            aom_list=error_aoms,
                            progress_callback=update_progress,
                        )

                        # Mettre à jour les résultats dans la session
                        # Remplacer les anciens résultats en erreur par les nouveaux
                        updated_results = []
                        for result in st.session_state.batch_results:
                            # Si c'était une AOM en erreur, chercher le nouveau résultat
                            if result.status == "error":
                                new_result = next(
                                    (
                                        r
                                        for r in results
                                        if r.n_siren_aom == result.n_siren_aom
                                    ),
                                    result,  # Garder l'ancien si pas trouvé
                                )
                                updated_results.append(new_result)
                            else:
                                # Garder les résultats qui n'étaient pas en erreur
                                updated_results.append(result)

                        st.session_state.batch_results = updated_results
                        st.success(
                            f"✅ Relancement terminé pour {len(error_aoms)} AOMs"
                        )
                        st.rerun()  # Recharger la page pour afficher les nouveaux résultats

                    except Exception as e:
                        st.error(f"❌ Erreur lors du relancement: {str(e)}")
                    finally:
                        st.session_state.batch_processing_active = False
        else:
            st.info("Aucune AOM en erreur à relancer")
