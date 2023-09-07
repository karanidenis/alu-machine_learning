import numpy as np
"""
This module contains the function:
performs matrix multiplication,
assume mat1 and mat2 are never empty and 
can be interpreted as numpy.ndarray
"""


def np_matmul(mat1, mat2):
    """
    matrix multiplication
    no loops ot conditions
    """
    mult = np.dot(mat1, mat2)
    return mult
