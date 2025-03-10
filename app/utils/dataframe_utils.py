import pandas as pd


def filter_dataframe(df: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """
    Filtre le DataFrame en fonction du terme de recherche.
    Args:
        df: DataFrame à filtrer
        search_term: Terme de recherche
    Returns:
        DataFrame filtré
    """
    if not search_term:
        return df

    df_str = df.astype(str).apply(lambda x: x.str.lower())
    search_terms = search_term.lower().split()
    mask = pd.Series([True] * len(df), index=df.index)
    for term in search_terms:
        term_mask = pd.Series([False] * len(df), index=df.index)
        for col in df_str.columns:
            term_mask |= df_str[col].str.contains(term, na=False, regex=True)
        mask &= term_mask
    return df[mask]
