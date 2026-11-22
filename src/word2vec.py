"""
le corpus est df["lemmas"], produit par add_lemma_columns dans preprocessing.py.

+ code d'éxécution dans scripts/ ; et résultats dans outputs/word2vec/

une seule fonction pour C-BOW et Skip-gram, avec sg en paramètre, (AC)
mes notes 
un word2vec = 3 lignes. → une fonction avec des parametres en boucle qui sort un excel. 
Graphique en 2D de projection des mots pour voir la proximité des mots. 
mots tests, de fréquence moyenne (AC), à partir de top_words(df["lemmas"])
ex — enfant / école / éducation 
Importance des hyperparamteres, fenetre de contexte, taille des vecteurs.

reproductibilité : fixer seed et workers=1
"""