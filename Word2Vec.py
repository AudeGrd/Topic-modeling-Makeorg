#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:32:27 2026

@author: jeannecizeau
"""

import pandas as pd
import spacy 
import fr_dep_news_trf
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Loading Spacy Package + adding specific words to stopwords ("IA" ??)
nlp = fr_dep_news_trf.load()
sup = {'faut'}
nlp.Defaults.stop_words.update(sup)


# Read the data 
data = pd.read_excel("MasterSmartData_IA_propositions.xlsx")

# Preprocessing + lemmatisation of the data with Spacy (using TRF because more precise)

def preprocess_text (text):
    doc = nlp(text.lower())
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return words

preprocessed_prop = [preprocess_text(proposition) for proposition in data["Proposition"]]


# Implementing Word2Vec on proposition words 
model = Word2Vec(sentences=preprocessed_prop, sg=1, vector_size = 100, window = 5, min_count = 1)
# (Hyper)parameters to tune (default values selected for now):
# sg = 0 for CBOW, 1 for Skip-Gram
# vector_size : dimensionality of word vectors 
# window : maximum distance between the current and predicted word within a sentence
# min_count : minimal length of word to take into account 
model.save("word2vec.model")


# Evaluating model 
try:
    sim = model.wv.similarity('mettre', 'logo')
    print(sim)
except KeyError as e:
    print(f"KeyError: {e}")


# Plot first results (using PCA to reduce dimension)
word_vectors = model.wv[model.wv.index_to_key]
pca = PCA(n_components=2)  
result = pca.fit_transform(word_vectors)

plt.figure(figsize=(10, 8))
plt.scatter(result[:, 0], result[:, 1])

results = list(model.wv.index_to_key)
for i, word in enumerate(results):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=12)

plt.title("Words Visualisation")
plt.grid()
plt.show()

