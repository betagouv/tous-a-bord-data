import logging
import os

import pandas as pd
from sqlalchemy import create_engine, inspect, text


def get_postgres_cs():
    """Build connection string from environment variables"""
    host = os.environ.get("POSTGRES_HOST", "postgres")
    db = os.environ.get("POSTGRES_DB", "postgres")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return f"postgresql://{user}:{password}@{host}:5432/{db}"


def load_aoms_data_from_db():
    """
    Upload clean data from the PostgreSQL database
    Returns:
        DataFrame pandas containing the AOMs data or None in case of error
    """
    try:
        engine = create_engine(get_postgres_cs())
        with engine.connect() as conn:
            check_query = text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'aoms'
                ) AND EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'passim_aoms'
                );
            """
            )
            tables_exist = conn.execute(check_query).scalar()
            if not tables_exist:
                return None
            query = text(
                """
                SELECT * FROM aoms
            """
            )
            df = pd.read_sql(query, conn)
            # Query to join AOMs and Passim data
            # query = text("""
            #     SELECT a.*, p.type_tarification, p.description_tarification,
            #            p.conditions_eligibilite, p.justificatifs, p.prix
            #     FROM aoms a
            #     LEFT JOIN passim_aoms p ON a.n_siren = p.siren_aom
            #     ORDER BY a.nom_aom
            # """)
            # df = pd.read_sql(query, conn)
            # Rename columns to match the expected format
            # column_mapping = {
            # }
            # df = df.rename(columns={k: v for k, v in column_mapping.items()
            # if k in df.columns})
            return df
    except Exception as e:
        print(
            "Erreur lors du chargement des données depuis "
            f"PostgreSQL: {str(e)}"
        )
        return None


def check_tables_exist(engine):
    """
    Check if the necessary tables exist in the database
    Args:
        engine: SQLAlchemy engine
    Returns:
        dict: Dictionary with the table names and their status (True/False)
    """
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        required_tables = ["aoms", "communes", "passim_aoms"]
        result = {table: table in tables for table in required_tables}
        return result
    except Exception as e:
        logging.error(f"Erreur lors de la vérification des tables: {str(e)}")
        return {table: False for table in ["aoms", "communes", "passim_aoms"]}
