import os

import psycopg2
import streamlit as st
from pgvector.psycopg2 import register_vector


# Configuration de la connexion
def get_database_connection():
    conn = psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        database=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    register_vector(conn)
    return conn


# Test de la connexion
try:
    conn = get_database_connection()
    st.success("Connexion à la base de données réussie!")
    conn.close()
except Exception as e:
    st.error(f"Erreur de connexion à la base de données: {str(e)}")

st.title("Application RAG avec Streamlit et PostgreSQL")
