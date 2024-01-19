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
        self.L = len(layers)
        # self.cache - dict of all intermediary values of the network
        self.cache = {}

        # weights - dict of all weights and biases of the network
        self.weights = {}
        for l in range(self.L):
            if l == 0:
                self.weights['W1'] = np.random.randn(
                    layers[0], nx) * np.sqrt(2 / nx)
                self.weights['b1'] = np.zeros([layers[0], 1])

            else:
                self.weights['W{}'.format(l+1)] = np.random.randn(
                    layers[l], layers[l-1]) * np.sqrt(2. / layers[l-1])

                self.weights['b{}'.format(l+1)] = np.zeros((layers[l], 1))
