#!/usr/bin/env python3
import numpy as np
"""
This module contains the function:
concatenates two matrices along an axis
assume mat1 and mat2 are never empty and 
can be interpreted as numpy.ndarray
"""


def np_cat(mat1, mat2, axis=0):
    """
    function to concatenate two matrices along an axis
    """

    axis_mat = np.concatenate((mat1, mat2), axis=axis)
    # axis1_mat = np.concatenate((mat1, mat2.T), axis = 1)

    return axis_mat
