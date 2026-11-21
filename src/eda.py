"""Exploratory analysis functions: summary tables and figures.

This module contains code only: it computes results, it does not store them.

Usage from a local notebook (notebooks are not versioned):
    import sys; sys.path.append("..")
    from src.preprocessing import load_data, add_clean_columns
    from src import eda
    df = add_clean_columns(load_data())
    eda.plot_exposure(df)
"""

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from .preprocessing import TEXT_COLUMN, VOTES_COLUMN, VOTE_SHARE_COLUMNS

CHARACTER_LIMIT = 140


# ---------------------------------------------------------------------------
# Table overview
# ---------------------------------------------------------------------------

def column_summary(df):
    """One row per column: type, missing values and number of distinct values."""
    return pd.DataFrame({
        "dtype": df.dtypes,
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(1),
        "n_unique": df.nunique(),
    })


# ---------------------------------------------------------------------------
# Author metadata (optional variables: age, postal code)
# ---------------------------------------------------------------------------

def plot_age(df, bins=30):
    """Histogram of author age (missing values excluded)."""
    ages = df["Âge"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(ages, bins=bins, color="steelblue")
    ax.set_xlabel("Author age")
    ax.set_ylabel("Number of proposals")
    ax.set_title(f"Author age (n = {len(ages)}, missing = {df['Âge'].isna().sum()})")
    return fig


def department_counts(df):
    """Number of proposals per department (first 2 digits of the postal code).

    Requires the postal code normalized by preprocessing.load_data.
    Note: Corsica (2A/2B) appears as "20".
    """
    departments = df["Code postal"].dropna().str[:2]
    return departments.value_counts().rename_axis("department").rename("count").to_frame()


def plot_departments(df, n=20):
    """Bar chart of the departments with the most proposals."""
    counts = department_counts(df)["count"].head(n)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(counts.index, counts.values, color="steelblue")
    ax.set_xlabel("Department")
    ax.set_ylabel("Number of proposals")
    ax.set_title(f"Top {n} departments (postal code known for {df['Code postal'].notna().sum()} proposals)")
    return fig


# ---------------------------------------------------------------------------
# Votes
# ---------------------------------------------------------------------------

def vote_share_summary(df):
    """Describe vote shares and check that they sum to 1 (up to rounding)."""
    shares = df[VOTE_SHARE_COLUMNS]
    return shares.assign(total=shares.sum(axis=1)).describe().round(3)


def plot_exposure(df, bins=40):
    """Histogram of the number of votes per proposal."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(df[VOTES_COLUMN], bins=bins, color="steelblue")
    ax.set_xlabel("Number of votes per proposal")
    ax.set_ylabel("Number of proposals")
    ax.set_title("Exposure: votes per proposal")
    return fig


def plot_support_vs_exposure(df):
    """Share of 'for' votes against the number of votes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df[VOTES_COLUMN], df["% pour"], s=10, alpha=0.5)
    ax.set_xlabel("Number of votes")
    ax.set_ylabel("Share of 'for' votes")
    ax.set_title("Support vs exposure")
    return fig


def plot_votes_by_order(df):
    """Number of votes against the row index (file order)."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(df.index, df[VOTES_COLUMN], s=8, alpha=0.5)
    ax.set_xlabel("Row index (file order)")
    ax.set_ylabel("Number of votes")
    ax.set_title("Votes by file order")
    return fig


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

def length_summary(df):
    """Length of proposals in characters (raw and cleaned) and in words.

    Requires the columns added by preprocessing.add_clean_columns.
    """
    lengths = pd.DataFrame({
        "n_chars_raw": df[TEXT_COLUMN].str.len(),
        "n_chars": df["text_clean"].str.len(),
        "n_words": df["text_clean"].str.split().str.len(),
    })
    return lengths.describe().round(1)


def plot_length(df, limit=CHARACTER_LIMIT, bins=40):
    """Histogram of raw proposal length, with the platform character limit."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(df[TEXT_COLUMN].str.len(), bins=bins, color="steelblue")
    ax.axvline(limit, color="red", linestyle="--", label=f"{limit}-character limit")
    ax.set_xlabel("Characters per proposal (raw)")
    ax.set_ylabel("Number of proposals")
    ax.legend()
    return fig


def word_counts(token_lists):
    """Count word occurrences over a series of token lists."""
    return Counter(token for tokens in token_lists for token in tokens)


def vocabulary_summary(token_lists, min_count=5):
    """Corpus size, vocabulary size, hapax and words above a frequency threshold."""
    counts = word_counts(token_lists)
    n_vocab = len(counts)
    n_hapax = sum(1 for c in counts.values() if c == 1)
    n_frequent = sum(1 for c in counts.values() if c >= min_count)
    return pd.Series({
        "n_tokens": sum(counts.values()),
        "vocabulary_size": n_vocab,
        "n_hapax": n_hapax,
        "pct_hapax": round(100 * n_hapax / n_vocab, 1),
        f"n_words_count_ge_{min_count}": n_frequent,
        f"pct_words_count_ge_{min_count}": round(100 * n_frequent / n_vocab, 1),
    })


def compare_vocabularies(corpora, min_count=5):
    """Vocabulary summary side by side for several versions of the corpus.

    corpora: dict {name: series of token lists}, e.g. raw tokens vs lemmas.
    """
    return pd.DataFrame({
        name: vocabulary_summary(tokens, min_count).values
        for name, tokens in corpora.items()
    }, index=vocabulary_summary(next(iter(corpora.values())), min_count).index)


def top_words(token_lists, n=30, exclude=None):
    """Most frequent words with their counts.

    exclude: optional set of words to ignore (e.g. a stopword list).
    """
    counts = word_counts(token_lists)
    if exclude:
        counts = Counter({w: c for w, c in counts.items() if w not in exclude})
    return pd.DataFrame(counts.most_common(n), columns=["word", "count"])


def first_word_counts(token_lists, n=25):
    """Most frequent first words (the word right after "Il faut")."""
    first_words = pd.Series([tokens[0] for tokens in token_lists if tokens])
    return first_words.value_counts().head(n).rename("count").to_frame()
