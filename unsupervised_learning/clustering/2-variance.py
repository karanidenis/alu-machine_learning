#!/usr/bin/env python3

"""
This module contains a function that
calculates total intra-cluster variance for a dataset
"""

import numpy as np


def variance(X, C):
    """
    calculates intra-cluster variance for a dataset

    X: numpy.ndarray (n, d) containing the dataset that
    will be used for K-means clustering
        - n no. of data points
        - d no. of dimensions for each data point
    C: numpy.ndarray (k, d) containing the centroid
        for each cluster

    return:
        - var: total intra-cluster variance
    """
    dist = np.linalg.norm(X[:, None] - C, axis=-1)
    clss = np.argmin(dist, axis=-1)
    var = np.sum(np.linalg.norm(X - C[clss], axis=-1) ** 2)
    return var
