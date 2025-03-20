import io
import os
import re
import tempfile

import numpy as np
import pandas as pd
import requests
import streamlit as st
from constants.urls import URL_DATASET_AOM_DOC, URL_PASSIM

# from sentence_transformers import SentenceTransformer
from services.transport_gouv_client import get_aom_dataset
from sqlalchemy import create_engine, insert, inspect, text
from utils.db_utils import get_postgres_cs
from utils.parser_utils import format_column

st.set_page_config(
    page_title="Mise à jour de la BDD",
    page_icon="🔄",
)

# Créer l'engine SQLAlchemy
engine = create_engine(get_postgres_cs())


def process_transport_gouv_data(dataset_aoms):
    """Downloads and processes ODS file from dataset URL, handling both sheets.

    Args:
        dataset_aoms: Dataset containing the ODS file URL
    Returns:
        Tuple (AOM DataFrame, communes DataFrame) or (None, None) if error
    """
    try:
        url = dataset_aoms.get("url")
        if not url:
            st.error("No URL found in dataset")
            return None, None
        st.info("Downloading file...")
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Download error: {response.status_code}")
            return None, None
        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp_f:
            tmp_f.write(response.content)
            tmp_path = tmp_f.name
        try:
            excel_file = pd.ExcelFile(tmp_path, engine="odf")
            sheet_names = excel_file.sheet_names
            if len(sheet_names) < 2:
                st.error("File must contain two sheets (AOM and communes)")
                return None, None
            # Specify dtype for n_insee column as string
            dtype_specs = {"n_insee": str}
            dataset_aom = pd.read_excel(
                tmp_path,
                sheet_name=sheet_names[0],
                engine="odf",
                dtype=dtype_specs,
            )
            dataset_com = pd.read_excel(
                tmp_path,
                sheet_name=sheet_names[1],
                engine="odf",
                dtype=dtype_specs,
            )
            dataset_aom.columns = [
                format_column(col) for col in dataset_aom.columns
            ]
            dataset_com.columns = [
                format_column(col) for col in dataset_com.columns
            ]
            # Ensure n_insee is string type after column formatting
            if "n_insee" in dataset_aom.columns:
                dataset_aom["n_insee"] = dataset_aom["n_insee"].astype(str)
            if "n_insee" in dataset_com.columns:
                dataset_com["n_insee"] = dataset_com["n_insee"].astype(str)
            st.success(
                "File loaded successfully! Sheets: "
                f"{', '.join(sheet_names)}"
            )
            # Store dataframes in session state for later use
            st.session_state.aoms_df = dataset_aom
            st.session_state.communes_df = dataset_com
            return dataset_aom, dataset_com
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None


def process_uploaded_passim_csv(uploaded_file):
    """
    Process the uploaded file and return a DataFrame.
    Handles CSV files with different separators.
    Args:
        uploaded_file: The file uploaded via st.file_uploader
    Returns:
        DataFrame pandas or None in case of error
    """
    try:
        preview = uploaded_file.read(1024).decode("utf-8")
        st.write("Aperçu du fichier :")
        st.code(preview)
        uploaded_file.seek(0)
        separators = [";", ",", "\t", "|"]
        for sep in separators:
            try:
                df = pd.read_csv(
                    uploaded_file,
                    sep=sep,
                    encoding="utf-8",
                    on_bad_lines="warn",
                )
                df.columns = [format_column(col) for col in df.columns]
                st.success(
                    f"Fichier lu avec succès en utilisant"
                    f"le séparateur '{sep}'"
                )
                # Store dataframe in session state for later use
                st.session_state.passim_df = df
                return df
            except Exception as e:
                st.warning(
                    f"Tentative avec séparateur '{sep}'" f"échouée: {str(e)}"
                )
                uploaded_file.seek(0)
        st.error("Impossible de lire le fichier avec les séparateurs standard")
        return None
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
        return None


def populate_aom_table(df_aom, engine):
    """
    Populate aoms table with optimized COPY method.
    Args:
        df_aom: DataFrame with AOM data
        engine: SQLAlchemy engine
    Returns:
        bool: True if successful, False otherwise
    """
    st.info("Updating AOM table...")
    try:
        inspector = inspect(engine)
        db_col = [col["name"] for col in inspector.get_columns("aoms")]
        common_columns = [col for col in df_aom.columns if col in db_col]
        df_filtered = df_aom[common_columns]
        ignored_columns = [col for col in df_aom.columns if col not in db_col]
        if ignored_columns:
            st.info(f"Ignored columns in aoms: {', '.join(ignored_columns)}")
        with engine.connect() as conn:
            # Empty the table
            conn.execute(text("TRUNCATE TABLE aoms;"))
            conn.commit()
            # Convert DataFrame to CSV in memory buffer
            csv_buffer = io.StringIO()
            df_filtered.to_csv(
                csv_buffer, index=False, header=False, sep="\t", na_rep="\\N"
            )
            csv_buffer.seek(0)
            # Use COPY command for fast insertion
            cursor = conn.connection.cursor()
            cursor.copy_from(
                csv_buffer, "aoms", columns=common_columns, null="\\N"
            )
            conn.connection.commit()
            req_aom = text("SELECT COUNT(*) FROM aoms")
            aom_count = conn.execute(req_aom).scalar()
        st.success(f"AOM table updated successfully: {aom_count} records")
        return True
    except Exception as e:
        st.error(f"Error updating AOM table: {str(e)}")
        import traceback

        st.error(traceback.format_exc())
        return False


def populate_communes_table(df_communes, engine):
    """
    Populate communes table with optimized COPY method.
    Args:
        df_communes: DataFrame with communes data
        engine: SQLAlchemy engine
    Returns:
        bool: True if successful, False otherwise
    """
    st.info("Updating Communes table...")
    try:
        inspector = inspect(engine)
        db_columns = [col["name"] for col in inspector.get_columns("communes")]
        column_types = {
            col["name"]: col["type"]
            for col in inspector.get_columns("communes")
        }
        # Create a copy of DataFrame to avoid modifying the original
        df_copy = df_communes.copy()
        numeric_columns = [
            col
            for col in df_copy.columns
            if col in column_types
            and hasattr(column_types[col], "python_type")
            and column_types[col].python_type in (int, float)
        ]
        for col in numeric_columns:
            # Replace problematic values with NaN
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].replace(
                    ["#N/D", "#N/A", "N/A", "NA", ""], np.nan
                )
                # Convert to appropriate type
                if column_types[col].python_type == int:
                    # For integer columns, we need to handle NaN specially
                    # First convert to float (which can hold NaN)
                    df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
                    # Then replace NaN with None for SQL NULL
                    df_copy[col] = df_copy[col].where(
                        pd.notna(df_copy[col]), None
                    )
                else:
                    # For float columns, pandas handles NaN correctly
                    df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
        common_columns = [col for col in df_copy.columns if col in db_columns]
        df_filtered = df_copy[common_columns]
        ignored_columns = [
            col for col in df_copy.columns if col not in db_columns
        ]
        if ignored_columns:
            st.info(
                f"Ignored columns in communes: {', '.join(ignored_columns)}"
            )
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE communes;"))
            conn.commit()
            try:
                df_filtered.to_sql(
                    "communes",
                    conn,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=1000,
                )
            except Exception as e:
                st.warning(
                    f"Fast insert failed, trying alternative method: {str(e)}"
                )
                insert_stmt = insert(text("communes")).values(
                    [
                        {col: row[col] for col in common_columns}
                        for _, row in df_filtered.iterrows()
                    ]
                )
                conn.execute(insert_stmt)
                conn.commit()
            req_communes = text("SELECT COUNT(*) FROM communes")
            communes_count = conn.execute(req_communes).scalar()
        st.success(
            f"Communes table updated successfully: {communes_count} records"
        )
        return True
    except Exception as e:
        st.error(f"Error updating Communes table: {str(e)}")
        import traceback

        st.error(traceback.format_exc())
        return False


def populate_passim_table(passim_df, engine):
    """
    Populate passim_aoms table with optimized COPY method.
    Args:
        passim_df: DataFrame with Passim data
        engine: SQLAlchemy engine
    Returns:
        bool: True if successful, False otherwise
    """
    st.info("Updating Passim table...")
    try:
        # Get table columns from database
        inspector = inspect(engine)
        db_columns = [
            col["name"] for col in inspector.get_columns("passim_aoms")
        ]
        # Mapping for Passim columns
        passim_mapping = {"fiche_transbus_tc": "fiche_transbus", "id": "_id"}
        df_copy = passim_df.copy()
        if "id" in df_copy.columns and "id" in db_columns:
            if not pd.to_numeric(df_copy["id"], errors="coerce").notna().all():
                st.warning(
                    "La colonne 'id' contient des valeurs non-numériques. "
                    "Génération de nouveaux IDs."
                )
                df_copy = df_copy.drop(columns=["id"])
        df_copy = df_copy.rename(
            columns={
                k: v for k, v in passim_mapping.items() if k in df_copy.columns
            }
        )
        # Clean text data - replace carriage returns, newlines and tabs
        for col in df_copy.columns:
            if df_copy[col].dtype == "object":
                df_copy[col] = (
                    df_copy[col]
                    .astype(str)
                    .apply(
                        lambda x: re.sub(r"[\r\n\t]", " ", x)
                        if pd.notna(x) and x != "nan"
                        else None
                    )
                )
        bool_columns = [
            col for col in df_copy.columns if df_copy[col].dtype == bool
        ]
        for col in bool_columns:
            df_copy[col] = df_copy[col].map(
                {True: "t", False: "f", None: "\\N"}
            )
        common_columns = [col for col in df_copy.columns if col in db_columns]
        if "id" in common_columns:
            common_columns.remove("id")
        df_filtered = df_copy[common_columns]
        ignored_columns = [
            col for col in df_copy.columns if col not in db_columns
        ]
        if ignored_columns:
            st.info(
                f"Ignored columns in passim_aoms: {', '.join(ignored_columns)}"
            )
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE passim_aoms;"))
            conn.commit()
            csv_buffer = io.StringIO()
            df_filtered.to_csv(
                csv_buffer, index=False, header=False, sep="\t", na_rep="\\N"
            )
            csv_buffer.seek(0)
            cursor = conn.connection.cursor()
            cursor.copy_from(
                csv_buffer, "passim_aoms", columns=common_columns, null="\\N"
            )
            conn.connection.commit()
            req_passim = text("SELECT COUNT(*) FROM passim_aoms")
            passim_count = conn.execute(req_passim).scalar()
        st.success(
            f"Passim table updated successfully: {passim_count} records"
        )
        return True
    except Exception as e:
        st.error(f"Error updating Passim table: {str(e)}")
        import traceback

        st.error(traceback.format_exc())
        return False


# Afficher des informations sur les données actuellement dans la base
st.header("Données actuellement dans la base")

with st.expander("Voir les données actuelles"):
    try:
        # Récupérer le nombre de lignes dans chaque table
        with engine.connect() as conn:
            query_AOM = text("SELECT COUNT(*) FROM aoms")
            query_communes = text("SELECT COUNT(*) FROM communes")
            query_passim = text("SELECT COUNT(*) FROM passim_aoms")
            aoms_count = conn.execute(query_AOM).scalar()
            communes_count = conn.execute(query_communes).scalar()
            passim_count = conn.execute(query_passim).scalar()

        # Table AOM
        st.subheader("Table AOMs")
        st.metric("Nombre d'enregistrements", aoms_count)
        if aoms_count > 0:
            aoms_sample = pd.read_sql("SELECT * FROM aoms LIMIT 5", engine)
            st.dataframe(aoms_sample)
        # Table Communes
        st.subheader("Table Communes")
        st.metric("Nombre d'enregistrements", communes_count)
        if communes_count > 0:
            communes_sample = pd.read_sql(
                "SELECT * FROM communes LIMIT 5", engine
            )
            st.dataframe(communes_sample)
        # Table Passim
        st.subheader("Table Passim")
        st.metric("Nombre d'enregistrements", passim_count)
        if passim_count > 0:
            passim_sample = pd.read_sql(
                "SELECT * FROM passim_aoms LIMIT 5", engine
            )
            st.dataframe(passim_sample)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {str(e)}")


st.header("🔄 Mise à jour de la base de données")
st.markdown("##### Source des données : transport.gouv.fr")
st.markdown(
    """Les données des AOMs (Autorités Organisatrices de la Mobilité) et des
communes proviennent initialement du **CEREMA**
(Centre d'études et d'expertise sur les risques, l'environnement,
la mobilité et l'aménagement) et sont formattées
par **transport.gouv.fr**.

Cette base de données contient :
- La liste des AOM
- Leur périmètre géographique
- Les communes membres
- Les informations de contact
"""
)
with st.expander("En savoir plus sur les données des AOMs"):
    st.markdown(
        f"""
        Pour plus d'informations sur les Autorités Organisatrices de la
        Mobilité (AOM) et la composition de la base de données, consultez
        la [documentation officielle]({URL_DATASET_AOM_DOC}).
        """
    )
dataset_aoms = get_aom_dataset()
# show the url of most recent aoms dataset on transport.gouv
# st.write(dataset_aoms)

df_aom, df_communes = process_transport_gouv_data(dataset_aoms)
if df_aom is not None and df_communes is not None:
    st.write("Aperçu des données :")
    st.dataframe(df_aom.head(5))
    st.dataframe(df_communes.head(5))

# Section Passim
st.markdown("##### Source des données : Passim")
st.markdown(
    """Les données sur les offres de transport proviennent de l'annuaire
Passim du Cerema.
"""
)
# Passim data is uploaded in the next page
st.markdown(
    """
**Guide étape par étape :**
1. Cliquez sur le lien ci-dessous pour ouvrir le site dans un nouvel onglet
2. Accédez à la section AOM
3. Cliquez sur le bouton "Exporter"
4. Sélectionnez le format CSV
5. Importez le fichier téléchargé ci-dessous
"""
)
st.markdown(
    f"""
    <div style="text-align: center; margin: 20px 0;">
        <a href="{URL_PASSIM}" target="_blank"
            style="background-color: #4CAF50; color: white;
                    padding: 10px 20px; text-decoration: none;
                    border-radius: 5px; font-weight: bold;">
            Ouvrir Passim dans un nouvel onglet
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file_csv = st.file_uploader(
    "Importez le fichier téléchargé depuis Passim",
    type=["csv"],
)
if uploaded_file_csv is not None:
    passim_df = process_uploaded_passim_csv(uploaded_file_csv)
    if passim_df is not None:
        st.write("Aperçu des données :")
        st.dataframe(passim_df.head(5))

# Section pour charger les données dans PostgreSQL
st.header("Mise à jour des données")

has_aoms_data = "aoms_df" in st.session_state
has_communes_data = "communes_df" in st.session_state
has_passim_data = "passim_df" in st.session_state

if "update_performed" not in st.session_state:
    st.session_state.update_performed = False

st.subheader("Résumé des données disponibles")

col1, col2, col3 = st.columns(3)

with col1:
    if has_aoms_data:
        st.success(f"✅ Données AOM: {len(st.session_state.aoms_df)} lignes")
        if st.button("🔄 AOM", key="btn_aom"):
            if populate_aom_table(st.session_state.aoms_df, engine):
                st.success(
                    "✅ Données AOM chargées avec succès: "
                    f"{len(st.session_state.aoms_df)} lignes"
                )
                st.session_state.update_performed = True
                st.rerun()
            else:
                st.error("❌ Échec du chargement des données AOM")
    else:
        st.warning("❌ Données AOM: Non disponibles")

with col2:
    if has_communes_data:
        st.success(
            "✅ Données Communes: "
            f"{len(st.session_state.communes_df)} lignes"
        )
        if st.button("🔄 les Communes", key="btn_communes"):
            if populate_communes_table(st.session_state.communes_df, engine):
                st.success(
                    "✅ Données Communes chargées avec succès: "
                    f"{len(st.session_state.communes_df)} lignes"
                )
                st.session_state.update_performed = True
                st.rerun()
            else:
                st.error("❌ Échec du chargement des données Communes")
    else:
        st.warning("❌ Données Communes: Non disponibles")

with col3:
    if has_passim_data:
        st.success(
            "✅ Données Passim: " f"{len(st.session_state.passim_df)} lignes"
        )
        if st.button("🔄 Passim", key="btn_passim"):
            if populate_passim_table(st.session_state.passim_df, engine):
                st.success(
                    "✅ Données Passim chargées avec succès: "
                    f"{len(st.session_state.passim_df)} lignes"
                )
                st.session_state.update_performed = True
                st.rerun()
            else:
                st.error("❌ Échec du chargement des données Passim")
    else:
        st.warning("❌ Données Passim: Non disponibles")

if st.session_state.update_performed:
    st.session_state.update_performed = False
