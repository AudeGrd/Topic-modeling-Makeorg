"""Shared loading and text cleaning functions.

This module contains code only: no data and no results.

Preprocessing decisions (see report, method section):
- spaCy French medium model (the small one mislabels key verbs such as "interdire");
- lemmas are lowercased ("IA", "Ia", "ia" -> "ia");
- spaCy stopwords are removed, negations included (they change the stance, not the topic);
- "faut" / "falloir" are added to the stopwords; "ia" is kept;
- all parts of speech are kept (proposals are too short to filter further);
- only alphabetic tokens are kept (punctuation, numbers and elisions such as "l'" are dropped).
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "MasterSmartData_IA_propositions.xlsx"

TEXT_COLUMN = "Proposition"
VOTES_COLUMN = "Nb de votes"
# Vote shares are kept as proportions in [0, 1], as in the raw file.
VOTE_SHARE_COLUMNS = ["% pour", "% contre", "% neutre"]

SPACY_MODEL = "fr_core_news_md"
# Added to spaCy's stopword list; compared with both the token and its lemma.
EXTRA_STOPWORDS = {"faut", "falloir"}

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


@lru_cache(maxsize=1)
def load_spacy(model=SPACY_MODEL):
    """Load the spaCy model once (parser and named entities are not needed)."""
    import spacy

    return spacy.load(model, disable=["parser", "ner"])


def keep_token(token, remove_stopwords=True):
    """Decide whether a spaCy token is kept in the preprocessed text."""
    if not token.is_alpha:
        return False
    if remove_stopwords:
        lemma = token.lemma_.lower()
        if token.is_stop or token.lower_ in EXTRA_STOPWORDS or lemma in EXTRA_STOPWORDS:
            return False
    return True


def lemmatize_texts(texts, remove_stopwords=True, nlp=None):
    """Turn each text into a list of lowercase lemmas.

    texts: iterable of cleaned texts (output of clean_text).
    Returns a list of token lists, in the same order as texts.
    """
    nlp = nlp or load_spacy()
    return [
        [token.lemma_.lower() for token in doc if keep_token(token, remove_stopwords)]
        for doc in nlp.pipe(texts, batch_size=64)
    ]


def add_lemma_columns(df):
    """Add lemmas with and without stopwords (requires add_clean_columns first).

    - "lemmas_all": every alphabetic lemma (to measure the effect of lemmatization alone);
    - "lemmas": lemmas without stopwords (the corpus used for Word2Vec).
    """
    df = df.copy()
    texts = df["text_clean"].tolist()
    df["lemmas_all"] = lemmatize_texts(texts, remove_stopwords=False)
    df["lemmas"] = lemmatize_texts(texts, remove_stopwords=True)
    return df
