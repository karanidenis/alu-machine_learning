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
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # initialise correlation matrix
    # cor = np.zeros((d, d))
    # print(cor)
    # print(cor.shape)
    # # diagonal elements are 1
    # np.fill_diagonal(cor, 1)
    # print(cor)
    # print(cor.shape)
    # # iterate over rows
    # for i in range(d):
    #     # iterate over columns
    #     for j in range(d):
    #         # calculate correlation
    #         cor[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])
    # return cor

    # initialise correlation matrix
    std_dev = np.sqrt(np.diag(C))
    # print(std_dev)
    # print(std_dev.shape)

    #  calculate correlation matrix
    cor = C / np.outer(std_dev, std_dev)
    # print(cor)
    return cor
