import json
import os
from typing import Dict, List

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
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
            max_tokens=4096,
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
            max_tokens=4096,
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
    # Étape 1: Concaténation du contenu
    with st.expander("1️⃣ Concaténation du contenu"):
        st.write("Assemblage de toutes les pages scrapées...")
        content = get_aom_content(selected_aom)
        st.text_area(
            "Contenu assemblé", value=content, height=200, disabled=True
        )
    # Étape 2: Analyse avec Claude
    with st.expander("2️⃣ Analyse du contenu avec Claude"):
        if st.button("Lancer l'analyse"):
            with st.spinner("Analyse en cours..."):
                analysis = analyze_content_with_claude(content)
                if analysis:  # Vérifier que l'analyse n'est pas None
                    # Afficher le score global
                    st.metric(
                        "Score global de pertinence",
                        format_score(analysis["score_global"]),
                    )
                    # Afficher les scores détaillés
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Score tarifs",
                            format_score(
                                analysis["details_score"]["presence_tarifs"]
                            ),
                        )
                    with col2:
                        st.metric(
                            "Score conditions",
                            format_score(
                                analysis["details_score"][
                                    "presence_conditions"
                                ]
                            ),
                        )
                    with col3:
                        st.metric(
                            "Score tarif solidaire",
                            format_score(
                                analysis["details_score"][
                                    "presence_tarif_solidaire"
                                ]
                            ),
                        )
                    # Afficher les sources les plus pertinentes
                    st.markdown("### Sources les plus pertinentes")
                    for url in analysis["sources_pertinentes"]:
                        st.write(f"- {url}")
                    # Afficher l'analyse détaillée
                    st.markdown("### Analyse détaillée")
                    st.write(analysis["analyse_detaillee"])

    # Ajouter la gestion des mots-clés
    if "available_keywords" not in st.session_state:
        st.session_state.available_keywords = default_keywords.copy()
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = default_keywords.copy()

    # Champ pour ajouter des mots-clés
    new_keyword = st.text_input(
        "Ajouter un nouveau mot-clé :",
        placeholder="Entrez un nouveau mot-clé et appuyez sur Entrée",
    )
    if new_keyword:
        if new_keyword not in st.session_state.available_keywords:
            st.session_state.available_keywords.append(new_keyword)
            st.session_state.selected_keywords.append(new_keyword)
            st.rerun()

    # Sélection des mots-clés
    selected_keywords = st.multiselect(
        "🏷️ Mots-clés pour l'analyse :",
        options=st.session_state.available_keywords,
        default=st.session_state.selected_keywords,
    )

    # Modifier l'expander d'analyse
    with st.expander("1️⃣ Extraction des passages pertinents"):
        st.write("Recherche des passages contenant les mots-clés...")
        content = get_aom_content(selected_aom)
        relevant_passages = extract_relevant_passages(
            content, selected_keywords
        )

        for url, passages in relevant_passages.items():
            if passages:
                st.markdown(f"**Source : {url}**")
                for passage in passages:
                    st.markdown(f"- {passage}")

    # Modifier l'interface pour afficher les informations tarifaires
    with st.expander("3️⃣ Extraction des tarifs"):
        if st.button("Extraire les tarifs"):
            with st.spinner("Extraction des tarifs en cours..."):
                tarifs = extract_tarif_info(content)
                if tarifs:
                    # Créer un DataFrame pour un affichage plus structuré
                    df = pd.DataFrame(tarifs)

                    # Trier par type de règle et montant
                    df = df.sort_values(["règle", "tarif"])

                    # Afficher le tableau
                    st.dataframe(
                        df,
                        column_config={
                            "règle": "Condition d'éligibilité",
                            "tarif": "Tarif (€)",
                            "unite": "Période",
                            "zone": "Zones",
                            "reduction": "Réduction",
                        },
                        hide_index=True,
                    )

                    # Ajouter un bouton pour télécharger les données
                    st.download_button(
                        "💾 Télécharger les tarifs (JSON)",
                        data=json.dumps(tarifs, ensure_ascii=False, indent=2),
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

            # Mettre à jour la barre de progression
            progress_bar.progress((idx + 1) / len(aoms))

    # Afficher le tableau de bord des scores
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
