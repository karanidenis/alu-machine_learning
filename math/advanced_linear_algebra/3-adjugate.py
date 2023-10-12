#!/usr/bin/env python3

"""
this module has the function adjugate(matrix)
"""


def adjugate(matrix):
    """
    calculates the adjugate
    """
    if len(matrix) < 1:
        raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]
    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]
    if len(matrix) == 3:
        a = (matrix[1][1]*matrix[2][2]) - (matrix[1][2]*matrix[2][1])
        b = (matrix[1][0]*matrix[2][2]) - (matrix[1][2]*matrix[2][0])
        c = (matrix[1][0]*matrix[2][1]) - (matrix[1][1]*matrix[2][0])
        d = (matrix[0][1]*matrix[2][2]) - (matrix[0][2]*matrix[2][1])
        e = (matrix[0][0]*matrix[2][2]) - (matrix[0][2]*matrix[2][0])
        f = (matrix[0][0]*matrix[2][1]) - (matrix[0][1]*matrix[2][0])
        g = (matrix[0][1]*matrix[1][2]) - (matrix[0][2]*matrix[1][1])
        h = (matrix[0][0]*matrix[1][2]) - (matrix[0][2]*matrix[1][0])
        i = (matrix[0][0]*matrix[1][1]) - (matrix[0][1]*matrix[1][0])
    # result = [[a, -b, c], [-d, e, -f], [g, -h, i]]
    adjugated_matrix = [[a, -d, g], [-b, e, -h], [c, -f, i]]
    return adjugated_matrix