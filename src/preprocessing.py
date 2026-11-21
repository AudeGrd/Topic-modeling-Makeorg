"""Shared loading and text cleaning functions.

Stopwords and lemmatization have be added once the preprocessing decisions are made.
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "MasterSmartData_IA_propositions.xlsx"

TEXT_COLUMN = "Proposition"
VOTES_COLUMN = "Nb de votes"
# Vote shares are kept as proportions in [0, 1], as in the raw file.
VOTE_SHARE_COLUMNS = ["% pour", "% contre", "% neutre"]

# Lowercase letters, including French accented letters.
# Apostrophes split words: "l'ia" -> "l", "ia".
TOKEN_PATTERN = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ]+")


def normalize_postal_code(postal_codes):
    """Return postal codes as 5-character strings, restoring leading zeros.

    The raw file stores postal codes as numbers, so 01000 is read as 1000.0.
    Note: astype(str).str.zfill(5) does not work ("1000.0" already has 6 characters).
    Missing values stay missing.
    """
    numeric = pd.to_numeric(postal_codes, errors="coerce").astype("Int64")
    return numeric.astype("string").str.zfill(5)


def load_data(path=RAW_FILE):
    """Load the raw Make.org file and fix the postal code column."""
    df = pd.read_excel(path)
    if "Code postal" in df.columns:
        df["Code postal"] = normalize_postal_code(df["Code postal"])
    return df


def clean_text(text):
    """Normalize apostrophes and remove the leading "Il faut" imposed by the platform."""
    text = text.replace("’", "'").strip()
    text = re.sub(r"^il faut\s+", "", text, flags=re.IGNORECASE)
    return text


def tokenize(text):
    """Lowercase a text and split it into word tokens (letters only)."""
    return TOKEN_PATTERN.findall(text.lower())


def add_clean_columns(df):
    """Add the cleaned text and its tokens to a copy of the dataframe."""
    df = df.copy()
    df["text_clean"] = df[TEXT_COLUMN].apply(clean_text)
    df["tokens"] = df["text_clean"].apply(tokenize)
    return df