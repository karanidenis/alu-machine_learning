#!/usr/bin/env python3

"""This module has a function that
creates a bag of words embedding matrix"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix
    sentence - list of sentences to analyze
    vocab - list of vocab words for analysis
    returns embeddings, features
    embeddings shape - (s, f)
        s-no. of sentences in sentences
        f-mo. of features analyzed
    features - list of features for embeddings
    """
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(vocabulary=vocab)

    # Fit the vectorizer on the documents and
    # transform the documents into the BoW matrix
    embedding_matrix = vectorizer.fit_transform(sentences).toarray()

    # Get the feature names (words) from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    return embedding_matrix, feature_names
