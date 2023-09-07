#!/usr/bin/env python3
"""
This module contains the function:
concatenates two matrices along an axis
assume mat1 and mat2 are never empty and
can be interpreted as numpy.ndarray
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    function to concatenate two matrices along an axis
    """

    axis_mat = np.concatenate((mat1, mat2), axis=axis)
    return axis_mat
