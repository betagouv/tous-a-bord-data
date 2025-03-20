import jellyfish
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from thefuzz import fuzz
from utils.db_utils import get_postgres_cs
from utils.parser_utils import normalize_string


def compare_strings_multiple_methods(str1: str, str2: str) -> dict:
    """Compare two strings using multiple similarity methods.
    Args:
        str1: First string to compare
        str2: Second string to compare
    Returns:
        Dictionary containing similarity scores for each method
    """
    # Ensure strings are not None
    str1 = "" if pd.isna(str1) else str(str1)
    str2 = "" if pd.isna(str2) else str(str2)
    # If either string is empty, return zero similarity
    if not str1 or not str2:
        return {"fuzz_ratio": 0, "levenshtein": 0, "jaro_winkler": 0}
    max_len = max(len(str1), len(str2))
    levenshtein_dist = jellyfish.levenshtein_distance(str1, str2)
    return {
        "fuzz_ratio": fuzz.ratio(str1, str2),
        "levenshtein": 100 * (1 - levenshtein_dist / max_len),
        "jaro_winkler": 100 * jellyfish.jaro_winkler_similarity(str1, str2),
    }


def find_matches_traditional(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    seuil_fuzz: int = 90,
    seuil_levenshtein: int = 90,
    seuil_jaro: int = 90,
) -> pd.DataFrame:
    """Find matches using traditional string similarity methods.
    Args:
        df_source: Source dataframe containing nom_aom_norm column
        df_target: Target dataframe containing nom_autorite_norm column
        seuil_fuzz: Minimum threshold for fuzz ratio
        seuil_levenshtein: Minimum threshold for levenshtein similarity
        seuil_jaro: Minimum threshold for jaro-winkler similarity
    Returns:
        DataFrame with matching results and similarity scores
    """
    correspondances = []
    for nom_source in df_source["nom_aom_norm"].unique():
        meilleures_corresp = []
        for nom_target in df_target["nom_autorite_norm"].unique():
            scores = compare_strings_multiple_methods(nom_source, nom_target)
            if (
                scores["fuzz_ratio"] >= seuil_fuzz
                or scores["levenshtein"] >= seuil_levenshtein
                or scores["jaro_winkler"] >= seuil_jaro
            ):
                meilleures_corresp.append(
                    {
                        "nom_source": nom_source,
                        "nom_target": nom_target,
                        **scores,
                    }
                )
        if meilleures_corresp:
            # Sort by fuzz ratio (could also use weighted average)
            meilleure = max(meilleures_corresp, key=lambda x: x["fuzz_ratio"])
            correspondances.append(meilleure)
    return pd.DataFrame(correspondances)


def find_matches_embeddings(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    model_name: str = "camembert-base",
) -> pd.DataFrame:
    """Find matches using text embeddings and cosine similarity.
    Args:
        df_source: Source dataframe containing nom_aom_norm column
        df_target: Target dataframe containing nom_autorite_norm column
        model_name: Name of the sentence-transformer model to use
    Returns:
        DataFrame with matching results and similarity scores
    """
    model = SentenceTransformer(model_name)
    source_embeddings = model.encode(df_source["nom_aom_norm"].unique())
    target_embeddings = model.encode(df_target["nom_autorite_norm"].unique())
    similarities = cosine_similarity(source_embeddings, target_embeddings)
    correspondances = []
    noms_source = df_source["nom_aom_norm"].unique()
    noms_target = df_target["nom_autorite_norm"].unique()
    for i, nom_source in enumerate(noms_source):
        meilleur_idx = np.argmax(similarities[i])
        score = similarities[i][meilleur_idx]
        if score > 0.95:  # Cosine similarity threshold
            correspondances.append(
                {
                    "nom_source": nom_source,
                    "nom_target": noms_target[meilleur_idx],
                    "similarity_score": float(score),
                }
            )
    return pd.DataFrame(correspondances)


"""Main function to execute the matching pipeline."""
st.title("Data Processing Pipeline")

# Load data from PostgreSQL
engine = create_engine(get_postgres_cs())
df_aoms = pd.read_sql("SELECT * FROM aoms", engine)
df_passim = pd.read_sql("SELECT * FROM passim_aoms", engine)

# Normalize names
df_aoms["nom_aom_norm"] = df_aoms["nom_aom"].apply(normalize_string)
df_passim["nom_autorite_norm"] = df_passim["autorite"].apply(normalize_string)
st.write(df_aoms.head())
st.write(df_passim.head())
# Find matches using traditional methods
corresp_trad = find_matches_traditional(df_aoms, df_passim)

# Find matches using embeddings
corresp_emb = find_matches_embeddings(df_aoms, df_passim)

# Display results
st.write("Matches using traditional methods:")
st.dataframe(corresp_trad)

st.write("Matches using embeddings:")
st.dataframe(corresp_emb)

# Compare results
comparaison = pd.merge(
    corresp_trad[["nom_source", "nom_target"]].assign(methode_trad=True),
    corresp_emb[["nom_source", "nom_target"]].assign(methode_emb=True),
    on=["nom_source"],
    how="outer",
    suffixes=("_trad", "_emb"),
)

st.write("Comparison of methods:")
st.dataframe(comparaison)
