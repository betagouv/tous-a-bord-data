import json
import re
from io import StringIO

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from utils.db_utils import get_postgres_cs, load_urls_data_from_db

st.title("Qualification du contenu scrapé")

# Connect to the database
engine = create_engine(get_postgres_cs())

# Load the URLs data to get the AOM names
urls_data = load_urls_data_from_db()
aom_names = dict(zip(urls_data["n_siren_aom"], urls_data["nom_aom"]))

# Get the list of unique AOMs
with engine.connect() as conn:
    aoms = conn.execute(
        text(
            """
        SELECT DISTINCT t.n_siren_aom, t.url_source,
        a.nom_aom, a.population_aom
        FROM tarification_raw t
        LEFT JOIN aoms a ON t.n_siren_aom = a.n_siren_aom
        ORDER BY a.population_aom DESC
    """
        )
    ).fetchall()

# Create a dictionary to store the qualified data
if "qualified_data" not in st.session_state:
    st.session_state.qualified_data = {}


# Function to save the qualified data
def save_qualification(siren, url, data):
    key = f"{siren}_{url}"
    st.session_state.qualified_data[key] = data
    st.success(f"✅ Données sauvegardées pour SIREN {siren}")


# Select the AOM to visualize
selected_aom = st.selectbox(
    "Select an AOM to qualify:",
    options=[(aom[0], aom[1]) for aom in aoms],
    format_func=lambda x: f"{x[0]} - "
    f"{aom_names.get(x[0], 'Unknown name')} - {x[1]}",
)

if selected_aom:
    siren, url = selected_aom
    nom_aom = aom_names.get(siren, "Nom inconnu")
    st.subheader(f"Qualification du contenu pour {nom_aom}")
    key = f"{siren}_{url}"
    existing_data = st.session_state.qualified_data.get(key, {})
    col1, col2 = st.columns(2)
    with col1:
        has_tarification = st.checkbox(
            "Contient des informations sur la tarification",
            value=existing_data.get("has_tarification", False),
        )
        tarification_complete = st.checkbox(
            "Informations de tarification complètes",
            value=existing_data.get("tarification_complete", False),
        )
    with col2:
        has_tarification_solidaire = st.checkbox(
            "Contient des informations sur la tarification solidaire",
            value=existing_data.get("has_tarification_solidaire", False),
        )
        tarification_solidaire_complete = st.checkbox(
            "Informations de tarification solidaire complètes",
            value=existing_data.get("tarification_solidaire_complete", False),
        )
    st.subheader("Critères d'éligibilité")
    categories_input = st.text_area(
        "Catégories d'éligibilité (une par ligne)",
        value="\n".join(existing_data.get("categorie_eligibilite", [])),
        help="Exemples: Étudiant, Senior, Demandeur d'emploi, etc.",
    )
    categorie_eligibilite = [
        cat.strip() for cat in categories_input.split("\n") if cat.strip()
    ]
    regles_input = st.text_area(
        "Règles d'éligibilité (une par ligne)",
        value="\n".join(existing_data.get("regle_eligibilite", [])),
        help="Exemples: Quotient familial < 800€, Âge > 65 ans, etc.",
    )
    regle_eligibilite = [
        regle.strip() for regle in regles_input.split("\n") if regle.strip()
    ]
    notes = st.text_area(
        "Notes supplémentaires",
        value=existing_data.get("notes", ""),
        help="Ajoutez ici toute information complémentaire",
    )
    if st.button("💾 Sauvegarder la qualification"):
        data = {
            "n_siren_aom": siren,
            "nom_aom": nom_aom,
            "url_source": url,
            "has_tarification": has_tarification,
            "tarification_complete": tarification_complete,
            "has_tarification_solidaire": has_tarification_solidaire,
            "tarification_solidaire_complete": tarification_solidaire_complete,
            "categorie_eligibilite": categorie_eligibilite,
            "regle_eligibilite": regle_eligibilite,
            "notes": notes,
        }
        save_qualification(siren, url, data)
    st.subheader("Export des données qualifiées")

    if st.session_state.qualified_data:
        data_list = []
        for key, data in st.session_state.qualified_data.items():
            data_copy = data.copy()
            data_copy["categorie_eligibilite"] = json.dumps(
                data_copy["categorie_eligibilite"], ensure_ascii=False
            )
            data_copy["regle_eligibilite"] = json.dumps(
                data_copy["regle_eligibilite"], ensure_ascii=False
            )
            data_list.append(data_copy)
        df = pd.DataFrame(data_list)
        st.dataframe(df)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Télécharger les données qualifiées (CSV)",
            data=csv_buffer.getvalue(),
            file_name="qualification_tarification.csv",
            mime="text/csv",
        )
        if st.button(
            "🗑️ Réinitialiser toutes les données qualifiées", type="secondary"
        ):
            st.session_state.qualified_data = {}
            st.success("✅ Données réinitialisées")
            st.rerun()
    else:
        st.info("Aucune donnée qualifiée disponible pour l'export")

    with engine.connect() as conn:
        pages = conn.execute(
            text(
                """
            SELECT url_page, contenu_scrape
            FROM tarification_raw
            WHERE n_siren_aom = :siren AND url_source = :url
            ORDER BY id
        """
            ),
            {"siren": siren, "url": url},
        ).fetchall()
    if pages:
        st.subheader(f"Contenu pour {nom_aom} (SIREN {siren}) - {url}")
        domain_match = re.match(r"https?://([^/]+)", url)
        if domain_match:
            domain = domain_match.group(0)
            st.markdown(f"[🌐 Visiter le site web]({domain})")
        tabs = st.tabs([f"Page {i+1}" for i in range(len(pages))])
        for i, page in enumerate(pages):
            with tabs[i]:
                st.markdown(f"**URL**: [{page[0]}]({page[0]})")
                st.markdown(page[1])
    else:
        st.warning(
            f"Aucune page trouvée pour {nom_aom}"
            f" (SIREN {siren}) et l'URL {url}"
        )
