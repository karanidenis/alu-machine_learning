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
    n, d = x.shape
    if type(x) is not np.ndarray and len(x.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(x, axis=0)
    x = x - mean
    cov = np.matmul(x.T, x) / (n - 1)
    return mean, cov
