#!/usr/bin/env python3

"""this module has a function that calculates
cost of  a neural entwork with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates weights and biases of a NN using gradient descent
    Y - one-hot of shape (classes, m) has
    labels for data
        classes - no. of classes
        m - no. of data points
    weights - dict of weights & biases of a NN
    cache - dict of outputs of each layer of NN
    alpha - learning rate
    lambtha - L2 regularization param
    L - no. of layers of the network
    NN uses tanh activations but last layer uses softmax"""

    # w = (1 - alpha(lambtha/m)) * w - alpha( der(cost) / der(w) )

    m = Y[1]

    gradients = {}
    # Initialize the backpropagation
    current_activation = cache["A{}".format(L)]
    dZ = current_activation - Y  # Derivative for softmax

    for l in reversed(range(1, L + 1)):
        A_prev = cache["A{}".format(l-1)] if l > 1 else cache["A0"]
        # if l == 1:
        #     # For the first hidden layer,
        #     # the previous layer's activation is the input data
        #     A_prev = cache['A0']
        # else:
        #     # For other layers, use the previous layer's activation
        #     A_prev = cache["A{}".format(l-1)]

        # Check shapes
        assert dZ.shape[0] == weights["W{}".format(
            l)].shape[0], "Shape mismatch in dZ for layer {}".format(l)
        # assert A_prev.shape[1] == X_train.shape[1],
        # f"Shape mismatch in A_prev for layer {l}"

        W = weights["W{}".format(l)]
        dW = (1/m) * np.dot(dZ, A_prev.T) + (lambtha/m) * W
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        gradients["dW{}".format(l)] = dW
        gradients["db{}".format(l)] = db

        if l > 1:  # No need to compute dZ for the input layer
            W_prev = weights["W{}".format(l-1)]
            # Derivative for tanh
            dZ = np.dot(W.T, dZ) * (1 - np.power(A_prev, 2))

    for l in range(1, L + 1):
        weights["W{}".format(l)] = (1 - alpha * (lambtha / m)) * \
            weights["W{}".format(l)] - alpha * gradients["dW{}".format(l)]
        weights["b{}".format(l)] -= alpha * gradients["db{}".format(l)]

    return gradients
