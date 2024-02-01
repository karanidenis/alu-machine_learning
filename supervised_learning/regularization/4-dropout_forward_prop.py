#!/usr/bin/env python3

"""this module has a function that conducts
forward propagation using Dropout"""

import tensorflow as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    """forward propagation using dropout
    X - shape(nx, m) has input data
        nx - no. of input features
        m - no. of data points
    weights - dict of weights and biases
    L - no. of layers
    keep_prob - probability a node will be kept
    tanh activation excpet last layer - softmax
    return dict of outputs of each layer and dropout mask used
    on each layer"""

    # weighted sum
    # for i in range(L, 0, -1):
    #     weighted_sum = np.dot(weights['W'+ str(i)], X) + weights['b' + str(i)]
    #     # applying activation function - tanh / softmax
    #     A = 1/(1 + np.exp(-weighted_sum))

    # return A

    cache = {'A0': X}

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

    return cache
