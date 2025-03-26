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
                    WHERE table_name = 'transport_offers'
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
            return df
    except Exception as e:
        print(
            "Erreur lors du chargement des données depuis "
            f"PostgreSQL: {str(e)}"
        )
        return None


def load_urls_data_from_db():
    """
    Upload clean data from the PostgreSQL database
    Returns:
        DataFrame pandas containing the AOMs data or None in case of error
    """
    try:
        engine = create_engine(get_postgres_cs())
        with engine.connect() as conn:
            query = text(
                """
                SELECT t.n_siren_aom,
                a.nom_aom,
                t.site_web_principal,
                t.nom_commercial,
                t.exploitant,
                t.type_de_contrat,
                a.population_aom,
                a.nombre_membre_aom,
                a.surface_km_2,
                t.type_d_usagers_faibles_revenus,
                t.type_d_usagers_recherche_d_emplois
                FROM transport_offers t
                LEFT JOIN aoms a ON t.n_siren_aom = a.n_siren_aom
                WHERE t.n_siren_aom IS NOT NULL
                AND t.site_web_principal IS NOT NULL
                ORDER BY a.population_aom DESC
                """
            )
            df = pd.read_sql(query, conn)
            return df
    except Exception as e:
        print(
            "Erreur lors du chargement des données depuis "
            f"PostgreSQL: {str(e)}"
        )


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
        required_tables = ["aoms", "communes", "transport_offers"]
        result = {table: table in tables for table in required_tables}
        return result
    except Exception as e:
        logging.error(f"Erreur lors de la vérification des tables: {str(e)}")
        return {
            table: False for table in ["aoms", "communes", "transport_offers"]
        }
