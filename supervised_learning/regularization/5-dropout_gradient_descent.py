#!/usr/bin/env python3

"""this module has a function that updates
weights of a NN with dropout regularization
using gradient descent"""

import numpy as np
import tensorflow as tf

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
    
    # cache = {'A0': X}

    for i in range(L):
        A = cache['A' + str(i)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        Z = tf.matmul(W, A) + b

        if i == L - 1:
            A = tf.nn.softmax(Z)
        else:
            A = tf.nn.tanh(Z)
            if keep_prob < 1.0:
                # Apply dropout
                D = tf.cast(tf.random.uniform(tf.shape(A))
                            < keep_prob, dtype=tf.float32)
                A = tf.math.multiply(A, D)
                A = A / keep_prob
                cache['D' + str(i + 1)] = D

        cache['A' + str(i + 1)] = A

    # return cache

    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        if i == L:
            dZ = A - Y
        else:
            dZ = dA * (1 - (A ** 2))
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + ((lambtha / m) * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        weights['W' + str(i)] = W - (alpha * dW)
        weights['b' + str(i)] = b - (alpha * db)
    return weights
