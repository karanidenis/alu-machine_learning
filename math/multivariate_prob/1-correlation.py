#!/usr/bin/env python3

"""
This module contains a function that
calculates a correlation matrix.
"""
import numpy as np


def correlation(C):
    """
    calculate correlation matrix
    C - numpy.ndarray - covariance matrix, shape (d, d)
    d - int - number of dimensions
    """

    d, d = C.shape
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    # if len(C.shape) != 2:
    if np.ndim(C) != 2 and d != d:
        raise ValueError("C must be a 2D square matrix")

    cor = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cor[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])
    return cor
