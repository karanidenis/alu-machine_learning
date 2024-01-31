#!/usr/bin/env python3

"""this module has a function that calculates
cost of  a neural entwork with L2 regularization"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """returns cost of the network accounting 
    for L2 regularization
    cost - cost of network without L2
    lambtha regularization param
    weights - dict of weights & biases of the nn
    L - no. of layers in NN
    m - no. of data points used"""

    # cf = cost + lambda / 2*m * sum(//w//**2)
    l2_penalty = 0
    for layer in range(1, L+1):
        l2_penalty += (weights[f'W{layer}'] ** 2).sum()

    l2_cost = cost + (lambtha / (2*m)) * l2_penalty

    return l2_cost
