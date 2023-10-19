#!/usr/bin/env python3

"""
This module contains a class that represents
a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """
    class initialization
    """

    def __init__(self, data):
        """
        class constructor
        data - numpy.ndarray - shape (d, n)-
        containing the data set:
        n - int - number of data points
        d - int - number of dimensions in each data point
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
        self.mean = mean
        cov = np.matmul(data - mean, data.T) / (data.shape[1] - 1)
        self.cov = cov
