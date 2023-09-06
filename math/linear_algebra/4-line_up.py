#!/usr/bin/env python3

"""
The module adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """ function adds two arrays element-wise """
    new_matrix = []

    if isinstance(arr1, list) and isinstance(arr2, list):
        if len(arr1) == len(arr2):
            for i in range(len(arr1)):
                new_matrix.append(arr1[i] + arr2[i])
            return new_matrix
        else:
            return None
