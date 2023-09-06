#!/usr/bin/env python3

"""
This module contains the function that performs matrix multiplication:
assumption is all matrices are 2D and all elements are in same dimension
of th same type/shape
"""


def mat_mul(mat1, mat2):
    """
    function  that performs matrix multiplication:
    """
    num_cols = len(mat1[0])
    num_rows = len(mat2)

    if num_rows == num_cols:
        return [[sum(a * b for a, b in zip(x_row, y_col))
                 for y_col in zip(*mat2)]
                for x_row in mat1]

    else:
        return None
