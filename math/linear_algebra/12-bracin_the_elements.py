#!/usr/bin/env python3
"""
This module contains the function:
perfoms element-wise addition, subtraction,
multiplication and division.
return a tuple with:
    (sum, difference, product, quotient)
"""


def np_elementwise(mat1, mat2):
    """
    assume mat1 and mat2 can be iterpreted as numpy.ndarray
    no loops or conditions
    assume mat(s) are never empty
    """

    add = mat1 + mat2
    sub = mat1 - mat2
    mut = mat1 * mat2
    div = mat1 / mat2

    return (add, sub, mut, div)
