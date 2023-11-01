#!/usr/bin/env python3

"""
This module contains a function that
calculates the posterior probability
n patients tested for a disease
x - number of patients who develop side effects
p - probability of side effects following binomial distribution
"""
import numpy as np


def likelihood(x, n, p):
    """
    claculates the likelihood of obtaining
    p is a 1D numpy.ndarray
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p, np.ndarray) or len(p.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(p > 1) or np.any(p < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    fact_n = 1
    for i in range(1, n + 1):
        fact_n *= i

    fact_x = 1
    for i in range(1, x + 1):
        fact_x *= i

    fact_nx = 1
    for i in range(1, n - x + 1):
        fact_nx *= i

    likelihood = (fact_n / (fact_x * fact_nx)) * (p ** x) *\
        ((1 - p) ** (n - x))

    return likelihood
