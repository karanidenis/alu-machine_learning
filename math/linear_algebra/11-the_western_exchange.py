#!/usr/bin/env python3
# import numpy as np
"""
This module contains the function:
transposes matrix
"""


def np_transpose(matrix):
    """
    assume matrix is translated as numpy.ndarray
    """
    if len(matrix) > 0:
        new_matrix = matrix.transpose
        return new_matrix
    else:
        return []