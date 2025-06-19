import pandas as pd


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
    # Préserver certains mots spécifiques avant la mise en minuscules
    special_words = {
        r"\bBus\b": "bus",
        r"\bDeux-roues\b": "deux_roues",
    }
    for pattern, replacement in special_words.items():
        column_name = re.sub(
            pattern, replacement, column_name, flags=re.IGNORECASE
        )
    column_name = column_name.lower()
    column_name = column_name.replace("_-_", "_")
    column_name = column_name.replace("(s)", "s")

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
    # Préserver les caractères spéciaux entre parenthèses comme (s)
    column_name = re.sub(r"\(s\)", "_s", column_name)
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
    # Do not singularize certain specific words
    exceptions_singularize = [
        "bus",
        "deux_roues_motorises",
        "usagers",
        "territoires",
    ]
    # Check if the name contains words that should not be singularized
    if any(exception in column_name for exception in exceptions_singularize):
        return column_name
    words = column_name.split("_")
    for i, word in enumerate(words):
        if word not in exceptions_singularize:
            # Singularize simple plural words
            if word.endswith("s") and len(word) > 1 and word[-2] != "s":
                words[i] = word[:-1]
    column_name = "_".join(words)
    return column_name


def normalize_string(s: str) -> str:
    """Normalize a string to improve matching quality.
    Args:
        s: Input string to normalize
    Returns:
        Normalized string with lowercase, no special chars and clean spaces
    """
    if not isinstance(s, str) or pd.isna(s):
        return s
    return (
        s.lower()
        .replace("-", " ")
        .replace("'", " ")
        .replace("/", " ")
        .replace("(", " ")
        .replace(")", " ")
        .strip()
        .replace("  ", " ")
    )
