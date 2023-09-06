#!/usr/bin/env python3

# def matrix_shape(matrix):
#     import numpy as np
#     return np.array(matrix).shape

def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None
    return shape
