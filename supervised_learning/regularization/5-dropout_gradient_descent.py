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
    cache - dict of outputs and dropout masks
    alpha - learning rate
    keep_prob - probability a node being kept
    L - no. of layers"""

    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        if i == L:
            dZ = A - Y
        else:
            dZ = dA * (1 - (A ** 2))
            dZ = np.multiply(dZ, cache['D' + str(i)])
            dZ /= keep_prob
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        weights['W' + str(i)] = W - (alpha * dW)
        weights['b' + str(i)] = b - (alpha * db)
    return weights
