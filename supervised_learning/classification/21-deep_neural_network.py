#!/usr/bin/env python3
"""This module is of a deep neural network
perfomaing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network"""

    def __init__(self, nx, layers):
        """class constructor
        nx - no. of input features
        layers - list of nodes in each layer of the network"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if (
            type(layers) is not list
            or len(layers) < 1
            or min(layers) < 1
        ):
            raise TypeError("layers must be a list of positive integers")

        # self.L - no. of layers in the neural network
        self.__L = len(layers)

        # self.cache - dict of all intermediary values of the network
        self.__cache = {}

        # weights - dict of all weights and biases of the network
        self.__weights = {}
        for layer in range(self.__L):
            if layer == 0:
                self.__weights['W1'] = np.random.randn(
                    layers[0], nx) * np.sqrt(2 / nx)
                self.__weights['b1'] = np.zeros([layers[0], 1])

            else:
                self.__weights['W{}'.format(layer+1)] = np.random.randn(
                    layers[layer],
                    layers[layer-1]) * np.sqrt(2. / layers[layer-1])

                self.__weights['b{}'.format(
                    layer+1)] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """calculate forward propagation of neural network
        X - contains input data. shape (nx, m)
        m - no. of examples"""
        # A = max(0, X)
        self.__cache['A0'] = X

        for l in range(1, self.__L+1):
            W = self.__weights['W{}'.format(l)]
            b = self.__weights['b{}'.format(l)]
            A_prev = self.__cache['A{}'.format(l-1)]

            Z = np.dot(W, A_prev) + b
            self.__cache['A{}'.format(l)] = 1 / (1 + np.exp(-Z))

        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """calculate cost of model using logistic regression
        Y - has correct labels for input data
        A - has activated output of the neuron"""
        m = Y.shape[1]
        cost = -(1/m) * np.sum([Y * np.log(A) +
                                (1 - Y) * np.log(1.0000001 - A)])
        return cost

    def evaluate(self, X, Y):
        """evaluate the neural network's prediction"""
        A, _ = self.forward_prop(X)
        # print(A.shape)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """calculate one pass of gradient descent on neural network
        alpha - learning rate"""
        m = Y.shape[1]
        A = self.__cache['A{}'.format(self.__L)]
        dz = A - Y

        for l in reversed(range(1, self.__L + 1)):
            A_prev = self.__cache['A{}'.format(l-1)]
            W = self.__weights['W{}'.format(l)]
            b = self.__weights['b{}'.format(l)]

            dW = (1 / m) * np.dot(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            # Update weights
            self.__weights['W{}'.format(l)] -= alpha * dW
            self.__weights['b{}'.format(l)] -= alpha * db

            if l > 1:
                A_prev = self.__cache['A{}'.format(l - 1)]
                dz = np.dot(W.T, dz) * A_prev * (1 - A_prev)

        return self.__weights
