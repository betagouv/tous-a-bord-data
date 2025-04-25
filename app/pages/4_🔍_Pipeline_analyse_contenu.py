import json
import os
from typing import Dict, List

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
from utils.db_utils import get_postgres_cs, load_urls_data_from_db

st.title("Pipeline d'analyse du contenu")

# Connect to the database
engine = create_engine(get_postgres_cs())

# After the imports
load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Ajouter les mots-clés par défaut
default_keywords = [
    "tarif",
    "abonnement",
    "solidaire",
    "pass",
    "titre",
    "ticket",
    "réduit",
    "jeune",
    "senior",
    "étudiant",
    "navigo",
]

# Constantes pour les modèles LLM disponibles
LLM_MODELS = {
    "Claude 3 Sonnet": {
        "name": "claude-3-5-sonnet-latest",
        "max_tokens": 200000,
    },
    "Claude 3 Haiku": {
        "name": "claude-3-5-haiku-latest",
        "max_tokens": 100000,
    },
}


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


def truncate_content(content, max_chars=100000):
    """Tronque le contenu en gardant le début et la fin"""
    if len(content) <= max_chars:
        return content

    half = max_chars // 2
    return content[:half] + "\n\n[...CONTENU TRONQUÉ...]\n\n" + content[-half:]


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
        truncated_content = truncate_content(content)
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
                    "content": f"{prompt}\n{truncated_content}",
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


def analyze_content_relevance(content, keywords):
    """
    Utiliser un LLM pour analyser la pertinence du contenu en fonction de:
    1. Présence de mots-clés importants
    2. Présence d'informations tarifaires (montants en euros)
    3. Identification des conditions d'éligibilité
    """
    f"""
    Analyser le contenu suivant et évaluer sa pertinence pour
    la tarification des transports.
    Critères à évaluer:
    1. Présence des mots-clés suivants: {', '.join(keywords)}
    2. Présence de montants en euros
    3. Présence de conditions d'éligibilité
    4. Présence d'informations sur la tarification solidaire
    Contenu à analyser:
    {content}
    Format de réponse attendu (JSON):
    {{
        "score": float, # entre 0 et 1
        "keywords_found": [str],
        "tarifs_identifies": [
            {{
                "montant": float,
                "unite": str,
                "description": str
            }}
        ],
        "conditions_eligibilite": [str],
        "has_tarification_solidaire": bool,
        "raisons": str
    }}
    """
    # Appel au LLM avec le prompt
    # response = llm.analyze(prompt)
    # return json.loads(response)


def extract_relevant_passages(
    content: str, keywords: List[str], context_chars: int = 200
) -> Dict[str, List[str]]:
    """Extrait les passages pertinents du contenu en fonction des mots-clés"""
    passages = {}

    # Découper le contenu en sections par URL
    sections = content.split("=== Source:")

    for section in sections:
        if not section.strip():
            continue

        # Extraire l'URL de la section
        url_line = section.split("\n")[0].strip()
        current_url = url_line.replace("===", "").strip()
        passages[current_url] = []

        # Pour chaque mot-clé, chercher les occurrences et leur contexte
        for keyword in keywords:
            start = 0
            while True:
                pos = section.lower().find(keyword.lower(), start)
                if pos == -1:
                    break

                # Extraire le contexte autour du mot-clé
                context_start = max(0, pos - context_chars)
                context_end = min(
                    len(section), pos + len(keyword) + context_chars
                )
                context = section[context_start:context_end]

                # Nettoyer et formater le passage
                context = context.replace("\n", " ").strip()
                context = (
                    f"...{context}..."
                    if len(context) == (context_chars * 2 + len(keyword))
                    else context
                )

                if context not in passages[current_url]:
                    passages[current_url].append(context)

                start = pos + len(keyword)

    return passages


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
    """Divide the content respecting the tokens limits of the model"""
    chunk_size = LLM_MODELS[model]["max_tokens"]
    chunks = []
    current_chunk = []
    current_size = 0
    # Split by pages to keep the consistency
    pages = content.split("--- Page:")
    for page in pages:
        page_size = len(page)
        if current_size + page_size <= chunk_size:
            current_chunk.append(page)
            current_size += page_size
        else:
            # Save the current chunk
            if current_chunk:
                chunks.append("--- Page:".join(current_chunk))
            # Start a new chunk
            current_chunk = [page]
            current_size = page_size
    # Add the last chunk
    if current_chunk:
        chunks.append("--- Page:".join(current_chunk))
    return chunks


def filter_content_by_relevance(
    content: str,
    keywords: List[str],
    model: str,
) -> Dict[str, str]:
    """Filtre le contenu pour ne garder que les parties pertinentes"""
    try:
        # Log pour debug
        st.write(
            f"Début du filtrage - "
            f"Taille du contenu: {len(content)} caractères"
        )
        # Diviser en chunks selon le modèle choisi
        chunks = split_content_in_chunks(content, model)
        st.write(f"Nombre de chunks: {len(chunks)}")
        filtered_chunks = []
        # Traiter chaque chunk
        for i, chunk in enumerate(chunks):
            st.write(f"Traitement du chunk {i+1}/{len(chunks)}")
            prompt = (
                "Filtrer le contenu suivant pour ne garder que les "
                "informations pertinentes.\n"
                "Ne conserver que:\n"
                "1. Les informations tarifaires (prix, montants en €)\n"
                "2. Les conditions d'éligibilité\n"
                "3. Les réductions et tarifs sociaux\n"
                "4. Les zones géographiques concernées\n\n"
                "IMPORTANT:\n"
                "- Garder uniquement les phrases pertinentes\n"
                "- Conserver la structure Source/Page\n"
                "- Ignorer tout contenu non lié aux tarifs\n"
                "- TOUJOURS retourner le contenu filtré, même minimal\n"
                "- Si aucune information pertinente n'est trouvée,\n"
                "  retourner 'Aucune information tarifaire trouvée'\n\n"
                f"Mots-clés importants: {', '.join(keywords)}\n\n"
                "Format de réponse attendu:\n"
                "- Garder les marqueurs '--- Page:' et leur contenu\n"
                "- Retourner le texte filtré directement"
            )

            # Utiliser le streaming avec max_tokens limité
            with client.messages.stream(
                model=LLM_MODELS[model]["name"],
                max_tokens=8000,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nContenu à filtrer:\n{chunk}",
                    }
                ],
            ) as stream:
                chunk_content = []
                for message in stream:
                    if message.type == "content_block":
                        chunk_content.append(message.text)
                filtered_text = "".join(chunk_content).strip()
                if filtered_text:  # Check that the text is not empty
                    filtered_chunks.append(filtered_text)
                    st.write(
                        f"✓ Chunk {i+1} filtré - "
                        f"Taille: {len(filtered_text)} caractères"
                    )
        # Combine the results
        combined_result = "\n\n".join(filtered_chunks)
        st.write(
            f"Filtrage terminé - "
            f"Taille du résultat: {len(combined_result)} caractères"
        )
        if not combined_result.strip():
            return {
                "Contenu filtré": "Pas d'info tarifaire pertinente trouvée"
            }
        return {"Contenu filtré": combined_result}
    except Exception as e:
        st.error(f"Erreur de filtrage : {str(e)}")
        st.write("Détails de l'erreur:", e)
        return {"Contenu filtré": f"Erreur lors du filtrage : {str(e)}"}


def aggregate_and_deduplicate(contents: Dict[str, str], model: str) -> str:
    """Agrège et déduplique le contenu de plusieurs sources"""
    try:
        all_content = "\n\n".join(contents.values())
        message = client.messages.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Agréger et dédupliquer le contenu suivant:
                1. Fusionner les informations similaires
                2. Éliminer les doublons
                3. Garder les informations les plus complètes
                4. Conserver la structure tarifaire

                Contenu:
                {all_content}""",
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        st.error(f"Erreur d'agrégation : {str(e)}")
        return ""


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
            model=model,
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
        st.session_state.available_keywords = default_keywords.copy()
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = default_keywords.copy()

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

            log_container = st.empty()
            progress_bar = st.progress(0)
            tabs = st.tabs(
                [
                    f"Source {i+1}"
                    for i in range(len(st.session_state.sources_content))
                ]
            )
            longest_source = max(
                st.session_state.sources_content.items(),
                key=lambda x: len(x[1]),
            )
            total_chunks = len(
                split_content_in_chunks(longest_source[1], selected_model)
            )
            # Traiter chaque source
            filtered_contents = {}
            for i, (source, content) in enumerate(
                st.session_state.sources_content.items()
            ):
                with tabs[i]:
                    st.write(f"URL: {source}")
                    # Afficher les informations de la source en cours
                    log_container.info(
                        f"Traitement de la source {i+1}: {source}"
                    )
                    log_container.write(
                        f"Taille du contenu: {len(content)} caractères"
                    )
                    chunks = split_content_in_chunks(content, selected_model)
                    log_container.write(
                        f"Nombre de chunks pour cette source: {len(chunks)}"
                    )
                    filtered_chunks = []
                    for j, chunk in enumerate(chunks):
                        # Mettre à jour le log pour le chunk en cours
                        log_container.info(
                            f"Filtrage du chunk {j+1}/{len(chunks)}"
                        )
                        with st.spinner(
                            f"Filtrage du chunk {j+1}/{len(chunks)}"
                        ):
                            prompt = f"""Filtrer le contenu suivant pour ne
                            garder que:
                            1. Les informations tarifaires
                            (prix, montants en €)
                            2. Les conditions d'éligibilité
                            3. Les réductions et tarifs sociaux
                            4. Les zones géographiques concernées
                            IMPORTANT:
                            - Garder uniquement les phrases pertinentes
                            - Conserver la structure Source/Page
                            - Ignorer tout contenu non lié aux tarifs,
                            critères d'éligibilité, réductions, etc.
                            Mots-clés importants:
                            {', '.join(selected_keywords)}
                            """

                            # Call Claude with streaming
                            with client.messages.stream(
                                model=LLM_MODELS[selected_model]["name"],
                                max_tokens=8000,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": (
                                            f"{prompt}\n\n"
                                            f"Contenu:\n{chunk}"
                                        ),
                                    }
                                ],
                            ) as stream:
                                chunk_content = []
                                for message in stream:
                                    if message.type == "content_block":
                                        chunk_content.append(message.text)
                                filtered_text = "".join(chunk_content).strip()
                                if filtered_text:
                                    filtered_chunks.append(filtered_text)
                                    log_container.write(
                                        f"✓ Chunk {j+1} filtré - "
                                        f"Taille: {len(filtered_text)}"
                                        "caractères"
                                    )
                        progress = (i * total_chunks + j + 1) / (
                            len(st.session_state.sources_content)
                            * total_chunks
                        )
                        progress_bar.progress(min(progress, 1.0))
                    # Combine the filtered chunks for this source
                    filtered_content = "\n\n".join(filtered_chunks)
                    filtered_contents[source] = filtered_content
                    # Display the filtered content in the tab
                    if filtered_content.strip():
                        st.text_area(
                            "Contenu filtré",
                            value=filtered_content,
                            height=300,
                            disabled=True,
                            key=f"filtered_content_{i}",
                        )
                    else:
                        st.warning(
                            "Aucun contenu pertinent trouvé dans cette source"
                        )
            # Save the filtered results for the next steps
            st.session_state.filtered_contents = filtered_contents
            # Final log
            log_container.success("Filtrage terminé pour toutes les sources")

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
