import json
import os
from typing import Dict, List

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from constants.keywords import DEFAULT_KEYWORDS
from dotenv import load_dotenv
from services.llm_services import call_anthropic, call_ollama, call_scaleway
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
    "Llama 3.1 8B Instruct FP16 (Ollama)": {
        "name": "llama3.1:8b-instruct-fp16",
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


def split_content_in_chunks(content: str, model: str) -> List[str]:
    """Découpe le contenu en chunks de taille max_tokens (en caractères)"""
    chunk_size = LLM_MODELS[model]["max_tokens"]
    # On découpe le texte brut, peu importe les pages
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append(content[start:end])
        start = end
    return chunks


def filter_content_by_relevance(
    content: str,
    keywords: List[str],
    model: str,
) -> Dict[str, str]:
    """Filtre le contenu pour ne garder que les parties pertinentes"""
    try:
        msg = (
            "Début du filtrage - "
            f"Taille du contenu: {len(content)} caractères"
        )
        st.write(msg)
        chunks = split_content_in_chunks(content, model)
        st.write(f"Nombre de chunks: {len(chunks)}")

        text_container = st.empty()
        filtered_content = ""

        for i, chunk in enumerate(chunks):
            st.write(f"Traitement du chunk {i+1}/{len(chunks)}")

            # Affichage du chunk pour debug
            st.info(
                f"--- Chunk {i+1} ---\n"
                f"Taille : {len(chunk)} caractères\n"
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

                    msg = (
                        f"✓ Chunk {i+1} filtré - "
                        f"Taille: {len(current_chunk_text)} caractères"
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
            msg = (
                f"Filtrage terminé - "
                f"Taille du résultat: {len(filtered_content)} caractères"
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


def aggregate_and_deduplicate(contents: Dict[str, str], model: str) -> str:
    """Agrège et déduplique le contenu de plusieurs sources"""
    all_content = "\n\n".join(contents.values())
    prompt = (
        "Agréger et dédupliquer le contenu suivant:\n"
        "1. Fusionner les informations similaires\n"
        "2. Éliminer les doublons\n"
        "3. Garder les informations les plus complètes\n"
        "4. Conserver la structure tarifaire\n\n"
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

    col1, col2 = st.columns([2, 1])
    with col1:
        new_keyword = st.text_input(
            "Ajouter un nouveau mot-clé :",
            placeholder="Entrez un nouveau mot-clé et appuyez sur Entrée",
            help="Le nouveau mot-clé sera ajouté à la liste disponible",
        )
    with col2:
        selected_model = st.selectbox(
            "Modèle LLM à utiliser :",
            options=list(LLM_MODELS.keys()),
            key="selected_llm",
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

        if st.button("Charger le contenu", key="load_content"):
            sources_content = {}
            tabs = st.tabs([f"Source {i+1}" for i in range(len(sources))])

            for i, source in enumerate(sources):
                with tabs[i]:
                    st.write(f"URL: {source}")
                    content = get_aom_content_by_source(selected_aom, source)
                    sources_content[source] = content
                    st.text_area(
                        "Contenu", value=content, height=300, disabled=True
                    )
            # Sauvegarder dans session_state pour les étapes suivantes
            st.session_state.sources_content = sources_content

    # Step 2: Filtrage du contenu
    st.header("🔍 Étape 2 : Filtrage du contenu")
    with st.expander("Filtrer le contenu pertinent"):
        if st.button("Lancer le filtrage", key="filter_content"):
            if "sources_content" not in st.session_state:
                st.error("Veuillez d'abord charger le contenu dans l'étape 1")
                st.stop()

            # Créer la barre de progression globale
            progress_bar = st.progress(0)

            # Créer les onglets
            sources_count = len(st.session_state.sources_content)
            tabs = st.tabs([f"Source {i+1}" for i in range(sources_count)])

            # Traiter chaque source
            filtered_contents = {}

            # Itérer sur les sources
            sources_items = st.session_state.sources_content.items()
            for i, (source, content) in enumerate(sources_items):
                with tabs[i]:
                    st.write(f"URL: {source}")

                    # Appeler la fonction de filtrage
                    filtered_result = filter_content_by_relevance(
                        content=content,
                        keywords=selected_keywords,
                        model=selected_model,
                    )

                    # Mettre à jour la barre de progression
                    progress_bar.progress((i + 1) / sources_count)

                    # Afficher le contenu filtré
                    if filtered_result["Contenu filtré"].strip():
                        st.text_area(
                            "Contenu filtré",
                            value=filtered_result["Contenu filtré"],
                            height=300,
                            disabled=True,
                            key=f"filtered_content_{i}",
                        )
                        filtered_contents[source] = filtered_result[
                            "Contenu filtré"
                        ]
                    else:
                        msg = (
                            "Aucun contenu pertinent trouvé dans cette source"
                        )
                        st.warning(msg)

            # Sauvegarder les résultats filtrés
            if filtered_contents:
                st.session_state.filtered_contents = filtered_contents
                st.success("Filtrage terminé pour toutes les sources")
            else:
                msg = "Aucun contenu pertinent n'a été trouvé dans les sources"
                st.error(msg)

    # Step 3: Agrégation et déduplication
    st.header("🔄 Étape 3 : Agrégation et déduplication")
    with st.expander("Agréger et dédupliquer"):
        if st.button("Lancer l'agrégation", key="aggregate_content"):
            if "filtered_contents" in st.session_state:
                aggregated_content = aggregate_and_deduplicate(
                    st.session_state.filtered_contents, selected_model
                )
                st.session_state.aggregated_content = aggregated_content
                st.text_area(
                    "Contenu agrégé",
                    value=aggregated_content,
                    height=300,
                    disabled=True,
                )

    # Step 4: Calcul du score cosinus
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
        if st.button("Structurer en JSON", key="structure_json"):
            if "aggregated_content" in st.session_state:
                structured_data = structure_content_as_json(
                    st.session_state.aggregated_content, selected_model
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
