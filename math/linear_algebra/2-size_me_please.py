#!/usr/bin/env python3

"""
This module contains a function for finding the shape of matrices.
"""


def matrix_shape(matrix):
    """ function finds the shape of matrices """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None
    return shape
