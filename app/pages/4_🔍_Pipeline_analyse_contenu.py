import json
import os
from typing import Dict, List

import pandas as pd
import streamlit as st
import tiktoken
from anthropic import Anthropic
from constants.keywords import DEFAULT_KEYWORDS
from dotenv import load_dotenv

# Nouveaux imports
from services.llm_services import call_anthropic, call_ollama, call_scaleway
from services.nlp_services import (
    extract_markdown_text,
    filter_text_with_spacy,
    load_spacy_model,
    normalize_text,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
from utils.db_utils import get_postgres_cs, load_urls_data_from_db

load_dotenv()

st.title("Pipeline d'analyse du contenu")

# Connect to the database
engine = create_engine(get_postgres_cs())

# After the imports
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# Constantes pour les modèles LLM disponibles
LLM_MODELS = {
    "Llama 3 (Ollama)": {
        "name": "llama3:8b",
        "max_tokens": 128000,
    },
    "Llama 3.3 70B (Scaleway)": {
        "name": "llama-3.3-70b-instruct",
        "max_tokens": 131000,
    },
    "Llama 3.1 8B (Scaleway)": {
        "name": "llama-3.1-8b-instruct",
        "max_tokens": 128000,
    },
    "Mistral Nemo (Scaleway)": {
        "name": "mistral-nemo-instruct-2407",
        "max_tokens": 128000,
    },
    "Qwen 2.5 32B (Scaleway)": {
        "name": "qwen2.5-coder-32b-instruct",
        "max_tokens": 32000,
    },
    # not really supported yet
    # "DeepSeek r1 (Scaleway)": {
    #     "name": "deepseek-r1",
    #     "max_tokens": 20000,
    # },
    "DeepSeek r1 distill (Scaleway)": {
        "name": "deepseek-r1-distill-llama-70b",
        "max_tokens": 32000,
    },
    "Claude 3 Haiku (Anthropic)": {
        "name": "claude-3-5-haiku-latest",
        "max_tokens": 100000,
    },
    # too expansive
    # "Claude 3 Sonnet (Anthropic)": {
    #     "name": "claude-3-5-sonnet-latest",
    #     "max_tokens": 200000,
    # },
}


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


def format_score(score: float) -> str:
    """Formate un score entre 0 et 1 en pourcentage limité à 100%"""
    return f"{min(score, 1.0):.2%}"


def get_aom_content(siren):
    """Concaténer tout le contenu d'un AOM pour toutes ses URLs sources"""
    with engine.connect() as conn:
        # D'abord récupérer toutes les URLs sources pour cet AOM
        urls = conn.execute(
            text(
                """
                SELECT DISTINCT url_source
                FROM tarification_raw
                WHERE n_siren_aom = :siren
            """
            ),
            {"siren": siren},
        ).fetchall()
        full_content = []
        for url in urls:
            # Pour chaque URL source, récupérer toutes les pages
            pages = conn.execute(
                text(
                    """
                    SELECT url_page, contenu_scrape
                    FROM tarification_raw
                    WHERE n_siren_aom = :siren AND url_source = :url
                    ORDER BY id
                """
                ),
                {"siren": siren, "url": url[0]},
            ).fetchall()
            # Ajouter un séparateur pour cette source
            full_content.append(f"\n\n=== Source: {url[0]} ===\n")
            # Ajouter le contenu de chaque page
            for page in pages:
                full_content.append(
                    f"--- Page: {page.url_page} ---\n{page.contenu_scrape}"
                )
    return "\n\n".join(full_content)


def extract_json_from_response(text):
    """Extrait le JSON de la réponse de Claude"""
    try:
        # Chercher le premier { et le dernier }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except Exception as e:
        st.error(f"Erreur d'extraction JSON : {str(e)}")
    return None


def analyze_content_with_claude(content):
    try:
        prompt = (
            "Analyser le contenu suivant en vous concentrant sur"
            "la tarification des transports en commun réguliers"
            "(bus, train, métro).\n"
            "IMPORTANT: Ne pas prendre en compte les services de transport"
            "à la demande (type PAM) sauf s'il n'y a pas d'autre information"
            "tarifaire.\n\n"
            "Critères d'analyse :\n"
            "1. Présence de tarifs standards pour les transports en commun\n"
            "2. Présence de tarifs réduits (jeunes, seniors, etc.)\n"
            "3. Présence de tarification solidaire (basée sur les revenus)\n"
            "4. Pertinence des sources (privilégier iledefrance-mobilites.fr)"
        )

        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n{content}",
                }
            ],
        )

        response_text = message.content[0].text
        result = extract_json_from_response(response_text)

        if not result:
            st.error("Impossible d'extraire le JSON de la réponse")
            st.text("Réponse brute de Claude :")
            st.text(response_text)
            return None

        return result

    except Exception as e:
        st.error(f"Erreur lors de l'appel à Claude : {str(e)}")
        return None


def extract_tarif_info(content: str) -> List[Dict]:
    """Extrait les informations tarifaires structurées du contenu"""
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-latest",
            messages=[
                {
                    "role": "user",
                    "content": f"""Extraire toutes les informations tarifaires
                du contenu suivant.
                Pour chaque tarif trouvé, structurer l'information dans une
                liste de dictionnaires avec ce format exact :
                {{
                    "règle": str,  # condition d'éligibilité
                    (ex: "moins de 11 ans", "AME", etc.)
                    "tarif": str,  # montant en euros
                    (format string avec virgule)
                    "unite": str,  # "an", "mois", "semaine"
                    "groupe": str,  # généralement "1"
                    "zone": str,  # zones concernées
                    (ex: "1 à 5", "2 à 3", etc.)
                    "reduction": str  # pourcentage de réduction
                    ("-50%", "-75%", "-100%", "")
                }}

                IMPORTANT:
                - Les montants doivent être en format string avec virgule
                (ex: "24,40")
                - Garder les règles exactes trouvées dans le texte
                - Inclure les zones géographiques si mentionnées
                - Indiquer les réductions en pourcentage avec le signe -
                - Laisser vide ("") si une information n'est pas disponible
                Contenu à analyser:
                {content}""",
                }
            ],
        )

        return json.loads(message.content[0].text)
    except Exception as e:
        st.error(f"Erreur d'extraction des tarifs : {str(e)}")
        return None


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
        content_parts = []
        for page in pages:
            content_parts.append(
                f"--- Page: {page.url_page} ---\n{page.contenu_scrape}"
            )
    return "\n\n".join(content_parts)


def get_tokenizer_for_model():
    """Retourne le bon tokenizer en fonction du modèle"""
    return tiktoken.get_encoding("cl100k_base").encode


def count_tokens(text: str) -> int:
    """Compte le nombre de tokens dans un texte (version générale)"""
    # Utiliser cl100k comme tokenizer général
    return len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text))


def split_content_in_chunks(content: str, model: str) -> List[str]:
    """Découpe le contenu en chunks de taille max_tokens"""
    max_tokens = LLM_MODELS[model]["max_tokens"]
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


def filter_content_by_relevance(
    content: str,
    keywords: List[str],
    model: str,
) -> Dict[str, str]:
    """Filtre le contenu pour ne garder que les parties pertinentes"""
    try:
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


def deduplicate_content(contents: Dict[str, str], model: str) -> str:
    """Deduplicate the content of multiple sources"""
    all_content = "\n\n".join(contents.values())
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
        "- Si tu ne trouves aucune information tarifaire, réponds "
        "Si l'information est en double, dedupliquer"
        "uniquement 'NO_TARIF_INFO'\n\n"
        f"Contenu:\n{all_content}"
    )
    result = select_model(model, prompt)
    return result


def compute_cosine_similarity(content: str, keywords: List[str]) -> float:
    """Calcule la similarité cosinus entre le contenu et les mots-clés"""
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode")
    # Préparer les textes
    texts = [content, " ".join(keywords)]
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Calculer la similarité
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


def structure_content_as_json(content: str, model: str) -> List[Dict]:
    """Structure le contenu au format JSON spécifié"""
    try:
        message = client.messages.create(
            model=LLM_MODELS[model]["name"],
            max_tokens=8000,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extraire et structurer les informations
                tarifaires du contenu suivant en JSON avec ce format exact:
                [{{
                    "règle": str,  # condition d'éligibilité
                    "tarif": str,  # montant avec virgule (ex: "24,40")
                    "unite": str,  # "an", "mois", "semaine"
                    "groupe": str,  # généralement "1"
                    "zone": str,  # zones (ex: "1 à 5")
                    "reduction": str  # "-50%", "-75%", "-100%", ""
                }}]

                Contenu:
                {content}""",
                }
            ],
        )
        return json.loads(message.content[0].text)
    except Exception as e:
        st.error(f"Erreur de structuration : {str(e)}")
        return []


# Interface Streamlit
st.subheader("Sélection de l'AOM à analyser")

# Load the URLs data to get the AOM names
urls_data = load_urls_data_from_db()
aom_names = dict(zip(urls_data["n_siren_aom"], urls_data["nom_aom"]))

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
    "Sélectionner un AOM:",
    options=[aom[0] for aom in aoms],
    format_func=lambda x: (
        f"{x} - "
        f"{next((a[1] for a in aoms if a[0] == x), 'Unknown')} "
        f"({next((a[2] for a in aoms if a[0] == x), 0)} sources)"
    ),
    key="selected_aom",
    on_change=lambda: (
        st.session_state.pop("all_content", None),
        st.session_state.pop("filtered_contents", None),
        st.session_state.pop("aggregated_content", None),
    ),
)

if selected_aom:
    nom_aom = next((a[1] for a in aoms if a[0] == selected_aom), "Nom inconnu")
    sources = next((a[3] for a in aoms if a[0] == selected_aom), "")
    st.write("Sources pour cet AOM:")
    for source in sources.split(" | "):
        st.write(f"- {source}")
    st.subheader("Pipeline d'analyse")
    # Step 0: Configuration des mots-clés
    st.header("🏷️ Étape 0 : Configuration des mots-clés")

    if "available_keywords" not in st.session_state:
        st.session_state.available_keywords = DEFAULT_KEYWORDS.copy()
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = DEFAULT_KEYWORDS.copy()

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
        "Mots-clés pour l'analyse :",
        options=st.session_state.available_keywords,
        default=st.session_state.selected_keywords,
    )

    # Step 1: Affichage du contenu scrapé
    st.header("📑 Étape 1 : Contenu scrapé")
    with st.expander("Afficher le contenu brut"):
        sources = next((a[3] for a in aoms if a[0] == selected_aom), "").split(
            " | "
        )

        sources_content = {}
        tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])

        # Concaténer le contenu de toutes les sources
        all_content = ""
        for i, source in enumerate(sources):
            with tabs[i]:
                st.write(f"URL: {source}")
                content = get_aom_content_by_source(selected_aom, source)
                sources_content[source] = content
                all_content += content + "\n\n"

        # Afficher le contenu dans les onglets
        for i, source in enumerate(sources):
            with tabs[i]:
                st.text_area(
                    "Contenu",
                    value=sources_content[source],
                    height=300,
                    disabled=True,
                )

        # Sauvegarder dans session_state pour les étapes suivantes
        st.session_state.all_content = all_content
        nb_tokens = count_tokens(all_content)
        st.write(f"Nombre de tokens : {nb_tokens}")

    # Step 2: Filtrage du contenu
    st.header("🔍 Étape 2 : Filtrage du contenu")
    with st.expander("Filtrer le contenu pertinent"):
        model_options = ["Filtrage NLP"] + list(LLM_MODELS.keys())
        selected_model_filter = st.selectbox(
            "Méthode de filtrage :",
            options=model_options,
            key="selected_llm_filter",
        )

        # Afficher le contenu filtré s'il existe
        if "filtered_contents" in st.session_state:
            filtered_content = st.session_state.filtered_contents[
                "Contenu filtré"
            ]
            if selected_model_filter == "Filtrage NLP":
                nb_tokens = count_tokens(
                    filtered_content
                )  # Version générale pour SpaCy
                title = f"Contenu filtré (SpaCy) - {nb_tokens} tokens"
            else:
                nb_tokens = count_tokens(filtered_content)
                title = f"Contenu filtré - {nb_tokens} tokens"
            st.text_area(
                title,
                value=filtered_content,
                height=300,
                disabled=True,
                key="filtered_content_display",
            )

        if st.button("Lancer le filtrage", key="filter_content"):
            # Vérification du contenu une seule fois
            all_content = st.session_state.get("all_content", {})
            if not all_content:
                st.error("Veuillez d'abord charger le contenu dans l'étape 1")
                st.stop()

            if selected_model_filter == "Filtrage NLP":
                # Chargement du modèle SpaCy une seule fois
                with st.spinner("Chargement du modèle SpaCy..."):
                    nlp = load_spacy_model()
                    raw_text = extract_markdown_text(all_content)
                    paragraphs = normalize_text(raw_text, nlp)
                    paragraphs_filtered, _ = filter_text_with_spacy(
                        paragraphs, nlp
                    )
                    filtered = "\n\n".join(paragraphs_filtered)

                    if filtered:
                        st.session_state.filtered_contents = {
                            "Contenu filtré": filtered
                        }
                        st.rerun()
                    else:
                        st.warning("Aucun contenu pertinent trouvé")
            else:
                filtered_result = filter_content_by_relevance(
                    content=all_content,
                    keywords=selected_keywords,
                    model=selected_model_filter,
                )

                if filtered_result["Contenu filtré"].strip():
                    st.session_state.filtered_contents = {
                        "Contenu filtré": filtered_result["Contenu filtré"]
                    }
                    st.success("Filtrage terminé")
                    st.rerun()
                else:
                    st.error("Aucun contenu pertinent trouvé dans les sources")

    # Step 3: Deduplication
    st.header("🔄 Étape 3 : Déduplication")
    with st.expander("Sélectionner le modèle LLM :"):
        selected_model_aggregate = st.selectbox(
            "Modèle LLM pour la déduplication :",
            options=list(LLM_MODELS.keys()),
            key="selected_llm_aggregate",
        )

        # Afficher le contenu agrégé s'il existe
        if "aggregated_content" in st.session_state:
            st.text_area(
                "Contenu agrégé",
                value=st.session_state.aggregated_content,
                height=300,
                disabled=True,
            )

        if st.button("Lancer la déduplication", key="dedup_content"):
            if "filtered_contents" in st.session_state:
                aggregated_content = deduplicate_content(
                    st.session_state.filtered_contents,
                    selected_model_aggregate,
                )
                st.session_state.aggregated_content = aggregated_content
                st.rerun()

    # Step 4: Compute similarity score
    st.header("📊 Étape 4 : Score de similarité")
    with st.expander("Calculer le score"):
        if st.button("Calculer le score", key="compute_score"):
            if "aggregated_content" in st.session_state:
                score = compute_cosine_similarity(
                    st.session_state.aggregated_content, selected_keywords
                )
                st.metric("Score de similarité", f"{score:.2%}")

    # Step 5: Structuration JSON
    st.header("🔧 Étape 5 : Structuration JSON")
    with st.expander("Structurer les données"):
        selected_model_structure = st.selectbox(
            "Modèle LLM pour la structuration :",
            options=list(LLM_MODELS.keys()),
            key="selected_llm_structure",
        )

        if st.button("Structurer en JSON", key="structure_json"):
            if "aggregated_content" in st.session_state:
                structured_data = structure_content_as_json(
                    st.session_state.aggregated_content,
                    selected_model_structure,
                )
                st.json(structured_data)
                # Download button
                st.download_button(
                    "💾 Download (JSON)",
                    data=json.dumps(
                        structured_data, ensure_ascii=False, indent=2
                    ),
                    file_name=f"tarifs_{selected_aom}.json",
                    mime="application/json",
                )

st.subheader("Analyse de la pertinence des contenus")

if st.button("Analyser tous les AOMs"):
    scores = {}
    progress_bar = st.progress(0)

    for idx, aom in enumerate(aoms):
        siren = aom[0]
        with st.spinner(f"Analyse de l'AOM {aom[1]} ({siren})..."):
            content = get_aom_content(siren)
            analysis = analyze_content_with_claude(content)

            if analysis:
                sources_pertinentes = analysis.get("sources_pertinentes", [])
                scores[siren] = {
                    "score": analysis.get("score_global", 0),
                    "best_page": {
                        "url": sources_pertinentes[0]
                        if sources_pertinentes
                        else None,
                        "analysis": {
                            "keywords_found": [],
                            "tarifs_identifies": [],
                        },
                    },
                }
            else:
                scores[siren] = {
                    "score": 0,
                    "best_page": {
                        "url": None,
                        "analysis": {
                            "keywords_found": [],
                            "tarifs_identifies": [],
                        },
                    },
                }

            # Update the progress bar
            progress_bar.progress((idx + 1) / len(aoms))

    # Display the score dashboard
    score_df = pd.DataFrame(
        [
            {
                "SIREN": siren,
                "Nom AOM": aom_names.get(siren, "Inconnu"),
                "Score": data["score"],
                "URL pertinente": data["best_page"]["url"],
                "Mots-clés trouvés": ", ".join(
                    data["best_page"]["analysis"]["keywords_found"]
                )
                if data["best_page"]
                else None,
                "Tarifs identifiés": len(
                    data["best_page"]["analysis"]["tarifs_identifies"]
                )
                if data["best_page"]
                else 0,
            }
            for siren, data in scores.items()
        ]
    )

    st.dataframe(score_df)
