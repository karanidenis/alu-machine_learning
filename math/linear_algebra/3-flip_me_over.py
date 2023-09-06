#!/usr/bin/env python3

"""
def matrix_transpose(matrix):
    import numpy as np
    return np.transpose(matrix)
"""


def matrix_transpose(matrix):
    """ Calculate the number of rows and columns in the input matrix """
    num_rows = len(matrix)
    # print(num_rows)
    num_cols = len(matrix[0]) if num_rows > 0 else 0
    # print(num_cols)

    # Create a new matrix to store the transpose
    transpose_matrix = [[0] * num_rows for _ in range(num_cols)]
    # print(transpose_matrix)

    # Populate the transpose matrix
    for i in range(num_rows):
        for j in range(num_cols):
            transpose_matrix[j][i] = matrix[i][j]

    return transpose_matrix
