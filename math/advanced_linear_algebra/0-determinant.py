#!/usr/bin/env python3

"""
module has the function determinant(matrix)
"""


def determinant(matrix):
    """
    calculate the determinant of a matrix
    """
    if matrix == [[]]:
        return 1
    if len(matrix) < 1:
    # and not isinstance(matrix[0], list):
            raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    if len(matrix) == 3:
        val1 = matrix[0][0]*matrix[1][1]*matrix[2][2]
        val2 = matrix[0][1]*matrix[1][2]*matrix[2][0]
        val3 = matrix[0][2]*matrix[1][0]*matrix[2][1]
        
        val4 =  matrix[0][2]*matrix[1][1]*matrix[2][0]
        val5 = matrix[0][1]*matrix[1][0]*matrix[2][2]
        val6 = matrix[0][0]*matrix[1][2]*matrix[2][1]
        
        result = (val1 + val2 + val3) - (val4 + val5 + val6)
        return result