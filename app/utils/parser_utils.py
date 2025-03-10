def format_column(column_name):
    """
    Formate un nom de colonne en minuscules,
    avec underscores comme séparateurs.
    - Détecte intelligemment les mots dans les chaînes CamelCase ou PascalCase
    - Traite correctement les sigles connus (AOM, SIREN, etc.)
    - Supprime les prépositions françaises (de, du, des, le, la, les)
    - Remplace les lettres accentuées par leurs équivalents sans accent
    - Gère correctement les apostrophes en les remplaçant par '_'
    - Convertit en minuscules
    - Remplace les espaces et autres séparateurs par des underscores
    - Supprime les caractères spéciaux

    Args:
        column_name: Le nom de colonne à formater
    Returns:
        Le nom de colonne formaté
    """
    import re
    import unicodedata

    column_name = str(column_name)
    # Handle special case for "N°"
    column_name = column_name.replace("N°", "n")
    column_name = unicodedata.normalize("NFKD", column_name)
    column_name = "".join(
        [c for c in column_name if not unicodedata.combining(c)]
    )
    # Handle CamelCase by inserting spaces
    column_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", column_name)
    # Insert spaces between letters and numbers
    column_name = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", column_name)
    column_name = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", column_name)
    # Handle special acronyms before lowercase
    column_name = re.sub(
        r"\bSIREN\s*Groupement\b",
        "siren_groupement",
        column_name,
        flags=re.IGNORECASE,
    )
    column_name = re.sub(
        r"\bSIRENAOM\b", "siren_aom", column_name, flags=re.IGNORECASE
    )
    column_name = re.sub(
        r"\bl[_ ]?AOM\b", "aom", column_name, flags=re.IGNORECASE
    )
    column_name = column_name.lower()
    # Remove French prepositions and articles using word boundaries
    prepositions = [
        r"\bl'",
        r"\bde\b",
        r"\bdu\b",
        r"\bdes\b",
        r"\ble\b",
        r"\bla\b",
        r"\bles\b",
        r"\bl\b",
        r"\bsur\b",
    ]
    for prep in prepositions:
        column_name = re.sub(f"{prep}", "", column_name)
    # Replace special chars and separators with spaces
    column_name = re.sub(
        r"[\'\/\\\-\.\,\:\;\|\+\#\&\%\(\)\[\]\{\}]", " ", column_name
    )
    # Remove any remaining special characters
    column_name = re.sub(r"[^a-z0-9_ ]", "", column_name)
    column_name = re.sub(r"\s+", " ", column_name)
    column_name = column_name.strip()
    # Clean up multiple spaces and convert to underscores
    column_name = re.sub(r"\s+", "_", column_name)
    column_name = re.sub(r"_+", "_", column_name)
    column_name = column_name.strip("_")
    # Handle special cases for SIREN at start
    condition = column_name.startswith("siren") and not any(
        x in column_name for x in ["groupement", "membre"]
    )
    if condition:
        column_name = "n_" + column_name
    # Handle numbers at start
    if column_name and column_name[0].isdigit():
        column_name = "col_" + column_name
    # Singularize simple plural words
    column_name = re.sub(r"([^s])s_", r"\1_", column_name)
    column_name = re.sub(r"([^s])s$", r"\1", column_name)
    return column_name
