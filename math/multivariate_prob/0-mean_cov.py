#!/usr/bin/env python3

"""
This module contains a function that
calculates the mean and covariance of a
data set.
"""

import numpy as np


def mean_cov(x):
    """
    calculate mean and covariance of a data set
    x - numpy.ndarray - data set, shape (n, d)
    n - int - number of data points
    d - int - number of dimensions
    """
    # x = np.array(x)
    # (n, d) = x.shape
    if not isinstance(x, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")

    # if n < 2:
    # if len(x.shape) != 2:
    if len(x.shape) < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(x, axis=0, keepdims=True)
    y = x - mean
    n = x.shape[0]
    cov = np.matmul(y.T, y) / (n - 1)
    # do not use np.matmul()1) # dot product
    return mean, cov
