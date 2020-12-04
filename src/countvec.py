import matplotlib.pyplot as plt
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text


my_stop_words = text.ENGLISH_STOP_WORDS.union(['hold','gonna','year','baby',"say", "thing", "like","know",
"get", "come", 'cause','time','yeah', 'nigga','niggas', 'niggaz', 'right'])


tfidf_vectorizer = TfidfVectorizer(stop_words=my_stop_words, max_features=7000)

count_vectorizer = CountVectorizer(stop_words=my_stop_words, max_features=7000)

def countvec(df, text = 'lyrics'):

    X = count_vectorizer.fit_transform(df[text])
    
    return X

def tfidf(df, text = 'lyrics'):

    X = tfidf_vectorizer.fit_transform(df[text])

    return X

def top_500_features(X, vec):
    
    features = count_vectorizer.get_feature_names()
    top_10_words = X.toarray().argsort()[:,-1:-501:-1]

    f = []

    for num, word in enumerate(top_10_words):
        f.append({', '.join(features[i] for i in word)})
    return f[0]

def tfidf_top_features(X, num):
    
    features = tfidf_vectorizer.get_feature_names()
    top_10_words = X.toarray().argsort()[:,-1:-(num+1):-1]

    f = []

    for num, word in enumerate(top_10_words):
        f.append({', '.join(features[i] for i in word)})
    return f

