#!/usr/bin/env python3

"""This module has a function that
creates a bag of words embedding matrix"""
# from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter


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
    # vectorizer = CountVectorizer(vocabulary=vocab)
     # Fit the vectorizer on the documents and
    # transform the documents into the BoW matrix
    # embedding_matrix = vectorizer.fit_transform(sentences).toarray()
    # Get the feature names (words) from the vectorizer
    # feature_names = vectorizer.get_feature_names_out()
    
    tokenized_sentences = []
    # Tokenize sentences
    for sentence in sentences:
        tokenized_sentences.append(sentence.lower().split())

    # for word in sentence:
    if vocab is None:
        vocab_set = set()
        for sentence in tokenized_sentences:
            for word in sentence:
                vocab_set.add(word)
        vocab = sorted(vocab_set)

    # word dictionary for vocab
    word_index = {}
    for i, word in enumerate(vocab):
        word_index[word] = i

    # initialize embedding matrix with zeros
    embeddings = np.zeros((len(sentences), len(vocab)),
                          dtype=int)

    # fill embedding matrix with wrd counts
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)
        for word, count in word_counts.items():
            if word in word_index:
                embeddings[i, word_index[word]] =  count
    return embeddings, vocab
