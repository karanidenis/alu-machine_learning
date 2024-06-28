#!/usr/bin/env python3

"""
This module contains a function that calculates
expectation step in the EM algorithm for a GMM
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    initializes variables for a Gaussian Mixture Model

    X: numpy.ndarray (n, d) containing the dataset
        - n no. of data points
        - d no. of dimensions for each data point
    pi: numpy.ndarray (k,) containing the priors for each cluster
    m: numpy.ndarray (k, d) containing centroid means for each cluster
    S: numpy.ndarray (k, d, d) covariance matrices for each cluster

    return:
        - g: numpy.ndarray (k, n) containing the posterior
            probabilities for each data point in each cluster
        -l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        return None, None
    if k != m.shape[0] or k != S.shape[0]:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None
    if not all(np.isclose(np.sum(pi), 1) for pi in S):
        return None, None
    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    g = g / np.sum(g, axis=0)
    l = np.sum(np.log(np.sum(g, axis=0)))
    return g, l
