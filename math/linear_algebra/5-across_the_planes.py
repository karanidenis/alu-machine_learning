#!/usr/bin/env python3

"""
This module contains the function adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """ function adds two matrices element-wise """

    new_matrix = []

    # check if matrix has more than list
    if len(mat1) == len(mat2):
        for arr1, arr2 in zip(mat1, mat2):
            if len(arr1) == len(arr2):
                new_matrix.append([x + y for x, y in zip(arr1, arr2)])
            else:
                return None
        return new_matrix
