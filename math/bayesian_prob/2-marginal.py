#!/usr/bin/env python3

"""
This module contains a function that
calculates the marginal probability of
obtaining the data
"""

import numpy as np
likelihood = __import__('0-likelihood').likelihood
intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """
    calculates the marginal probability of
    obtaining the data

    x--> number of patients that develop side effects
    n--> total number of patients observed
    P--> 1D numpy.ndarray containing the various hypothetical
         probabilities of developing side effects
    Pr--> 1D numpy.ndarray containing the prior beliefs of P

    Returns: the marginal probability of obtaining x and n
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not np.ndim(P) == 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not np.ndim(Pr) == 1:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.shape(P) == np.shape(Pr):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    #  formula for marginal probability:
    #  P(E) = P(E|H) * P(H)
    #  P(E|H) = likelihood
    #  P(H) = Pr

    return np.sum(likelihood(x, n, P) * Pr)
