# Topic Modeling & Opinion Mapping on a Make.org Citizen Consultation

> 🚧 **Work in progress**, ENSAI Master in Smart Data Science project (2026-2027)

Which themes emerge from a citizen consultation on artificial intelligence, and does the answer depend on the NLP method used? This project compares a **local NLP model**, **pre-trained sentence embeddings** and **direct LLM prompting** for unsupervised topic modeling, then maps the topics found to citizens' votes to locate areas of consensus and controversy.

## Context

[Make.org](https://make.org) runs large-scale citizen consultations: anyone can submit a proposal and vote on others' proposals. This project uses the consultation *"How can AI improve the lives of French citizens while limiting risks?"* (June to October 2025).

- **953 proposals** written in French, each starting with *"Il faut…"* and limited to 140 characters
- **~100,000 votes** (for / against / neutral)

There are no labels: as in most customer and citizen insight studies, the task is **unsupervised**.

## Research questions

1. What are the main themes of the consultation, and how do they differ depending on the method (Word2Vec, sentence embeddings or LLM prompting)?
2. How do approval rates (consensus vs. controversy) map onto the thematic clusters?

## Approach

| Step | Content | Status |
|---|---|---|
| 1. Exploratory analysis | Proposal length and vocabulary, vote distributions, most consensual and most divisive proposals; NLP preprocessing (tokenisation, normalisation, stop words, lemmatisation) | ✅ |
| 2. Word2Vec (local) | Model trained **on the corpus itself** rather than general-purpose pre-trained vectors, as the vocabulary and style are domain-specific. CBOW vs. Skip-gram, 20 to 100 dimensions, word-to-proposal aggregation (mean, TF-IDF weighting), clustering (e.g. K-Means), 2D projection (t-SNE / UMAP) | ⏳ |
| 3. Sentence embeddings | Pre-trained open-source multilingual / French encoders (`sentence-transformers`, Hugging Face), with the **same clustering** as step 2 for a fair comparison | ⏳ |
| 4. LLM prompting | Zero-shot / few-shot categorisation of a sample of proposals, compared with the vector-based clusters | ⏳ |
| 5. Evaluation & opinion mapping | See below | ⏳ |

## Evaluation

Without ground truth, no single metric can rank the methods. They are compared on:

- **human inspection**: thematic coherence of the proposals within each cluster, word clouds;
- **LLM-as-a-judge**: a model rates the homogeneity and clarity of each detected theme;
- **practical criteria**: topic granularity, computing time, implementation complexity, interpretability.

Each theme is then crossed with its approval rates to separate broadly supported topics from polarising ones.

## Data

The raw data belongs to Make.org and **is not shared in this repository**. Only code is published: no data, no results.

## Setup

```bash
pip install -r requirements.txt
nbstripout --install   # strips notebook outputs automatically before each commit
python -m spacy download fr_core_news_md
```

Place the Make.org data files in `data/raw/`. This folder is ignored by Git: never commit the data.

## Tech stack

Python, pandas, Gensim (Word2Vec), scikit-learn, sentence-transformers / Hugging Face, UMAP / t-SNE, an LLM API for steps 4 and 5. Versioned with Git and GitHub.

## Team

- Aude Guerand ([@AudeGrd](https://github.com/AudeGrd))
- Jeanne Cizeau ([@jeanneciz](https://github.com/jeanneciz))

Supervised by Anne-Cécile Gay, independent data scientist working with Make.org.
