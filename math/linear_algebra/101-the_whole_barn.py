#!/usr/bin/env python3
"""
This module has a function that adds two matrices
assume elements in the same dimension are of same type/shape
mat1 and 2 are never empty
"""
import numpy as np


def add_matrices(mat1, mat2):
    """
    addition while checking shape
    return None when shape is not same
    """

    mat1 = np.array(mat1)
    mat2 = np.array(mat2)

    if mat1.shape != mat2.shape:
        return None

    result = mat1 + mat2
    return result.tolist()
