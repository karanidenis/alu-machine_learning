#!/usr/bin/env python3

"""
This module contains the function that can concatenate 
two matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    fuction that can concatenate two matrices along a specific axis
    matrices are 2D represented as lists of lists
    """

    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None

    if axis == 0:
        new_matrix = mat1 + mat2
    elif axis == 1:
        new_matrix = [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        raise ValueError("Axis must be 0 or 1 for concatenation.")

    return new_matrix
