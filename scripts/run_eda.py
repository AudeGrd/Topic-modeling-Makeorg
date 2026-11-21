"""Run the full exploratory analysis and save tables and figures to outputs/eda/.

Usage, from the project root:
    python -m scripts.run_eda

Results are written to outputs/, which is ignored by Git: this script contains no results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt

from src import eda
from src.preprocessing import PROJECT_ROOT, add_clean_columns, load_data

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"


def save_table(table, name):
    table.to_csv(OUTPUT_DIR / f"{name}.csv", sep=";", encoding="utf-8-sig")


def save_figure(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    df = add_clean_columns(raw)

    # Table overview (on raw columns only)
    save_table(eda.column_summary(raw), "01_column_summary")

    # Votes
    save_table(eda.vote_share_summary(df), "02_vote_share_summary")
    save_figure(eda.plot_exposure(df), "02_exposure")
    save_figure(eda.plot_support_vs_exposure(df), "02_support_vs_exposure")
    save_figure(eda.plot_votes_by_order(df), "02_votes_by_order")

    # Text
    save_table(eda.length_summary(df), "03_length_summary")
    save_figure(eda.plot_length(df), "03_length")
    save_table(eda.vocabulary_summary(df["tokens"]).to_frame("value"), "03_vocabulary_summary")
    save_table(eda.top_words(df["tokens"]), "03_top_words")
    save_table(eda.first_word_counts(df["tokens"]), "03_first_words")

    print(f"EDA outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()