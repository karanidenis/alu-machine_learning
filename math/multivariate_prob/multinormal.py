#!/usr/bin/env python3

"""
This module contains a class that represents
a Multivariate Normal distribution.
"""

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