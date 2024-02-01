#!/usr/bin/env python3

"""this module has a function that updates
weights of a NN with dropout regularization
using gradient descent"""

import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update weigghts of a NN using gradient descent
    Y - one-hot of shape(classes, m) with correct data labels
        classes - no. of classes
        m - no. of data points
    weights - dict of weights and biases
    cache - dict of outputs and dropout masks"""