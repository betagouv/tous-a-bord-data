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
    create_mapping_matcher,
    extract_markdown_text,
    filter_transport_fare,
    load_spacy_model,
    normalize_text,
)
from sqlalchemy import create_engine, text
from star_ratings import star_ratings
from utils.crawler_utils import CrawlerManager
from utils.db_utils import get_postgres_cs

# Configuration de la page pour utiliser plus de largeur
st.set_page_config(
    page_title="Evaluation du traitement par AOM", layout="wide"
)

# Fonction pour lire le fichier bordeaux.txt
def load_example(type: str, aom_name: str) -> str:
    """Charge le contenu du fichier bordeaux.txt"""
    import os
    from pathlib import Path

    # Chemin vers le fichier bordeaux.txt
    current_dir = Path(__file__).parent.parent  # Remonter au dossier app
    aom_file = current_dir / "prompts" / "data" / type / f"{aom_name}.txt"

    try:
        with open(aom_file, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "Exemple non trouvé"
    except Exception as e:
        return f"Erreur lecture fichier: {str(e)}"


load_dotenv()

st.title("Evaluation du traitement par AOM")

# Connect to the database
engine = create_engine(get_postgres_cs())


# After the imports
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Constantes pour les modèles LLM disponibles


def select_model(model_name: str, prompt: str) -> str:
    """Select the model based on the model name"""
    if "Ollama" in model_name:
        current_chunk_text = call_ollama(
            prompt,
            model=LLM_MODELS[model_name]["name"],
        )
    elif "Anthropic" in model_name:
        current_chunk_text = call_anthropic(
            prompt, model=LLM_MODELS[model_name]["name"]
        )
    elif "Scaleway" in model_name:
        current_chunk_text = call_scaleway(
            prompt, model=LLM_MODELS[model_name]["name"]
        )
    else:
        raise ValueError(f"Modèle non supporté : {model_name}")
    return current_chunk_text


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


def get_tokenizer_for_model():
    """Retourne le bon tokenizer en fonction du modèle"""
    return tiktoken.get_encoding("cl100k_base").encode


def count_tokens(text: str) -> int:
    """Compte le nombre de tokens dans un texte (version générale)"""
    # Utiliser cl100k comme tokenizer général
    return len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text))


def split_content_in_chunks(content: str, model: str) -> List[str]:
    """Découpe le contenu en chunks de taille max_tokens"""
    max_tokens = LLM_MODELS[model]["max_tokens"] - MAX_TOKEN_OUTPUT * 2
    tokenizer = get_tokenizer_for_model()

    # Utiliser le même tokenizer pour l'encodage et le décodage
    try:
        decoder = tiktoken.get_encoding("cl100k_base").decode
    except KeyError:
        st.error("Erreur de décodage, utilisation de cl100k_base")
        decoder = tiktoken.get_encoding("cl100k_base").decode

    # Encoder le texte
    tokens = tokenizer(content)

    # Découper en chunks
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = decoder(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


# Initialiser le dictionnaire des run_ids s'il n'existe pas
if "run_ids" not in st.session_state:
    st.session_state.run_ids = {}


@traceable
def filter_content_by_relevance(
    content: str,
    keywords: List[str],
    model: str,
    siren: str,
    name: str,
) -> Dict[str, str]:
    """Filtre le contenu pour ne garder que les parties pertinentes"""
    try:
        run = get_current_run_tree()
        st.session_state.run_ids["filter"] = run.id

        nb_tokens = count_tokens(content)
        msg = "Début du filtrage - " f"Nombre de tokens: {nb_tokens}"
        st.write(msg)
        chunks = split_content_in_chunks(content, model)
        st.write(f"Nombre de chunks: {len(chunks)}")

        text_container = st.empty()
        filtered_content = ""

        for i, chunk in enumerate(chunks):
            st.write(f"Traitement du chunk {i+1}/{len(chunks)}")

            # Affichage du chunk pour debug
            chunk_tokens = count_tokens(chunk)
            st.info(
                f"--- Chunk {i+1} ---\n"
                f"Nombre de tokens : {chunk_tokens}\n"
                f"Contenu du chunk :\n{chunk[:2000]}"
            )

            prompt = (
                "Extrais toutes les informations tarifaires des transports en "
                "commun à partir du texte suivant. Garde exactement :\n"
                "1. Les tarifs standards (billets, carnets, abonnements)\n"
                "2. Les tarifs réduits et leurs conditions\n"
                "3. Les tarifs solidaires et leurs conditions d'éligibilité\n"
                "4. Les zones géographiques concernées\n\n"
                "IMPORTANT:\n"
                "- Conserve les montants exacts et les unités (€)\n"
                "- Garde toutes les conditions d'éligibilité\n"
                "- Ne fais pas de résumé ou d'interprétation\n"
                "- Retourne le texte brut avec sa structure\n"
                "- Si tu trouves des informations tarifaires, retourne-les\n"
                "- Ne retourne PAS de texte formaté ou de liste\n"
                "- Conserve les marqueurs '--- Page:' si présents\n\n"
                "- Si tu ne trouves aucune information tarifaire, réponds "
                "uniquement 'NO_TARIF_INFO'\n\n"
                f"Mots-clés de référence : {', '.join(keywords)}\n\n"
                "Contenu à filtrer :\n"
                f"{chunk}"
            )

            try:
                current_chunk_text = select_model(model, prompt)

                if "NO_TARIF_INFO" not in current_chunk_text:
                    if filtered_content:
                        filtered_content += "\n\n"
                    filtered_content += current_chunk_text

                    if filtered_content.strip():
                        text_container.text_area(
                            "Contenu filtré",
                            value=filtered_content,
                            height=300,
                            disabled=True,
                        )

                    chunk_tokens = count_tokens(current_chunk_text)
                    msg = (
                        f"✓ Chunk {i+1} filtré - "
                        f"Nombre de tokens: {chunk_tokens}"
                    )
                    st.write(msg)
                else:
                    st.warning(
                        f"⚠️ Chunk {i+1} : Aucune information tarifaire"
                    )

            except Exception as chunk_error:
                msg = (
                    f"Erreur lors du traitement du chunk {i+1}: "
                    f"{str(chunk_error)}"
                )
                st.warning(msg)
                continue

        if filtered_content:
            nb_tokens = count_tokens(filtered_content)
            msg = (
                f"Filtrage terminé - "
                f"Nombre de tokens du résultat: {nb_tokens}"
            )
            st.write(msg)
            return {"Contenu filtré": filtered_content}
        else:
            text_container.empty()
            return {"Contenu filtré": ""}

    except Exception as e:
        st.error(f"Erreur de filtrage : {str(e)}")
        st.write("Détails de l'erreur:", e)
        return {"Contenu filtré": f"Erreur lors du filtrage : {str(e)}"}


@traceable
def filter_content_with_nlp(
    content: str, model: str, siren: str, name: str
) -> Dict[str, str]:
    """Filtre le contenu avec SpaCy"""
    try:
        run = get_current_run_tree()
        st.session_state.run_ids["filter"] = run.id

        with st.spinner("Chargement du modèle SpaCy..."):
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
def pre_format(
    contents: Dict[str, str], model: str, siren: str, name: str
) -> str:
    """Nettoie le contenu pour ne garder que les informations tarifaires"""
    run = get_current_run_tree()
    st.session_state.run_ids["pre_format"] = run.id

    all_content = "\n\n".join(contents.values())
    max_tokens = LLM_MODELS[model]["max_tokens"]
    nb_tokens = count_tokens(all_content)
    prompt = (
        "Extrais toutes les informations tarifaires des transports en "
        "commun à partir du texte suivant. Garde exactement :\n"
        "1. Les tarifs standards (billets, carnets, abonnements)\n"
        "2. Les tarifs réduits et leurs conditions\n"
        "3. Les tarifs solidaires et leurs conditions d'éligibilité\n"
        "4. Les zones géographiques concernées\n\n"
        "IMPORTANT:\n"
        "- Conserve les montants exacts et les unités (€)\n"
        "- Garde toutes les conditions d'éligibilité\n"
        "- Ne fais pas de résumé ou d'interprétation\n"
        "- Retourne le texte brut avec sa structure\n"
        "- Si tu trouves des informations tarifaires, retourne-les\n"
        "- Ne retourne PAS de texte formaté ou de liste\n"
        "- Si l'information est en double, dedupliquer\n"
        "- Si tu ne trouves aucune information tarifaire, réponds "
        "uniquement 'NO_TARIF_INFO'\n\n"
    )
    # Créer un conteneur pour le résultat
    result_container = st.empty()
    if nb_tokens > max_tokens:
        chunks = split_content_in_chunks(all_content, model)
        st.write(f"Nombre de chunks : {len(chunks)}")
        extracted_parts = []
        for i, chunk in enumerate(chunks):
            st.write(f"Traitement du chunk {i+1}/{len(chunks)}")
            chunk_result = select_model(model, prompt + f"Contenu:\n{chunk}")
            if "NO_TARIF_INFO" not in chunk_result:
                extracted_parts.append(chunk_result)
                # Afficher le résultat partiel
                result_container.text_area(
                    "Résultat de l'extraction",
                    value="\n\n".join(extracted_parts),
                    height=300,
                    disabled=True,
                )
        # Fusionner les résultats des chunks
        final_result = "\n\n".join(extracted_parts)
        # Afficher le résultat final
        result_container.text_area(
            "Résultat final de l'extraction",
            value=final_result,
            height=300,
            disabled=True,
        )
        return final_result
    else:
        result = select_model(model, prompt + f"Contenu:\n{all_content}")
        # Afficher le résultat
        result_container.text_area(
            "Résultat de l'extraction", value=result, height=300, disabled=True
        )
        return result


@traceable
def format_publicode(content: str, model: str, siren: str, name: str) -> str:
    """Convertit le contenu en format Publicode"""
    run = get_current_run_tree()
    st.session_state.run_ids["format_publicode"] = run.id

    # Charger l'exemple de Bordeaux
    aom_name = "bordeaux"
    example_tsst = load_example("tsst", aom_name)
    example_publicode = load_example("publicode", aom_name)
    prompt = text_to_publicode(
        example_tsst,
        example_publicode,
        content,
    )
    return select_model(model, prompt)


@traceable
def format_tags(text: str, nlp, siren: str, name: str) -> List[str]:
    """Extrait les tags uniques à partir du texte"""
    run = get_current_run_tree()
    st.session_state.run_ids["format_tags"] = run.id
    # Créer le matcher
    matcher = create_mapping_matcher(nlp)

    # Traiter le texte
    doc = nlp(text)

    # Trouver les correspondances et récupérer les tags uniques
    matches = matcher(doc)
    tags = set()
    for _, start, end in matches:
        token_span = doc[start:end]
        token_lemma = token_span[
            0
        ].lemma_.lower()  # Utiliser le lemme du premier token
        # Chercher la clé correspondante dans le mapping
        for mapping_key, mapping_value in TAG_DP_MAPPING.items():
            # Comparer les lemmes
            mapping_doc = nlp(mapping_key.lower())
            if mapping_doc[
                0
            ].lemma_.lower() == token_lemma and mapping_value.get("tag"):
                tags.add(mapping_value["tag"])
                break

    return sorted(list(filter(None, tags)))


@traceable
def format_data_providers(text: str, nlp, siren: str, name: str) -> List[str]:
    """Extrait les fournisseurs de données uniques à partir du texte"""
    run = get_current_run_tree()
    st.session_state.run_ids["format_data_providers"] = run.id
    # Créer le matcher
    matcher = create_mapping_matcher(nlp)

    # Traiter le texte
    doc = nlp(text)

    # Trouver les correspondances et récupérer les fournisseurs uniques
    matches = matcher(doc)
    providers = set()
    for _, start, end in matches:
        token_span = doc[start:end]
        token_lemma = token_span[
            0
        ].lemma_.lower()  # Utiliser le lemme du premier token
        # Chercher la clé correspondante dans le mapping
        for mapping_key, mapping_value in TAG_DP_MAPPING.items():
            # Comparer les lemmes
            mapping_doc = nlp(mapping_key.lower())
            if mapping_doc[
                0
            ].lemma_.lower() == token_lemma and mapping_value.get(
                "fournisseur"
            ):
                providers.add(mapping_value["fournisseur"])
                break

    return sorted(list(filter(None, providers)))


def show_evaluation_interface(step_name: str, content: str) -> None:
    """Affiche l'interface d'évaluation pour une étape"""
    st.divider()
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
        st.session_state.pop("pre_formatted_content", None),
        st.session_state.pop("publicode", None),
        st.session_state.pop("tags", None),
        st.session_state.pop("netext_content", None),
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

        model_options = ["Filtrage NLP"] + list(LLM_MODELS.keys())
        selected_model_filter = st.selectbox(
            "Méthode de filtrage :",
            options=model_options,
            key="selected_llm_filter",
            disabled=not is_previous_step_complete,
        )

        # Afficher le contenu filtré s'il existe
        if "filtered_contents" in st.session_state:
            filtered_content = st.session_state.filtered_contents[
                "Contenu filtré"
            ]
            nb_tokens = count_tokens(filtered_content)
            if selected_model_filter == "Filtrage NLP":
                title = f"Contenu filtré (SpaCy) - {nb_tokens} tokens"
            else:
                title = f"Contenu filtré - {nb_tokens} tokens"
            st.text_area(
                title,
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
            if selected_model_filter == "Filtrage NLP":
                filtered_result = filter_content_with_nlp(
                    scraped_content,
                    selected_model_filter,
                    n_siren_aom,
                    nom_aom,
                )
                if filtered_result["Contenu filtré"].strip():
                    st.session_state.filtered_contents = filtered_result
                    st.success("Filtrage terminé")
                    st.rerun()
                else:
                    st.error("Aucun contenu pertinent trouvé dans les sources")
            else:
                filtered_result = filter_content_by_relevance(
                    content=scraped_content,
                    keywords=selected_keywords,
                    model=selected_model_filter,
                    siren=n_siren_aom,
                    name=nom_aom,
                )

                if filtered_result["Contenu filtré"].strip():
                    st.session_state.filtered_contents = filtered_result
                    st.success("Filtrage terminé")
                    st.rerun()
                else:
                    st.error("Aucun contenu pertinent trouvé dans les sources")

    # Step 4: Extraction des tags

    with st.expander("Étape 4 : Extraction des tags", expanded=True):
        is_previous_step_complete = (
            "filtered_contents" in st.session_state
            and st.session_state.filtered_contents.get(
                "Contenu filtré", ""
            ).strip()
        )

        if not is_previous_step_complete:
            st.warning("⚠️ Veuillez d'abord compléter l'étape de filtrage")
        if st.button(
            "Extraire les tags",
            key="format_in_tags",
            use_container_width=True,
            disabled=not is_previous_step_complete,
        ):
            with st.spinner("Génération des tags en cours..."):
                # Charger le modèle SpaCy
                nlp = load_spacy_model()
                # Extraire les tags et data providers
                st.session_state.tags = format_tags(
                    st.session_state.filtered_contents[
                        "Contenu filtré"
                    ].strip(),
                    nlp,
                    n_siren_aom,
                    nom_aom,
                )
                st.rerun()

        # Afficher les tags s'ils existent dans la session
        if "tags" in st.session_state:
            st.session_state.tags = st_tags(
                label="# Tags détectés :",
                text="",
                value=st.session_state.tags,
                key="tag_display",
            )
            show_evaluation_interface("format_tags", st.session_state.tags)
