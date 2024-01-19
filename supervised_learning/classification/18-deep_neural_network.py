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

        # if not isinstance(layers, list) or len(layers) < 1:
        #     raise TypeError("layers must be a list of positive integers")
        # if not all(isinstance(layer, int) and layer > 0 for layer in layers):
        #     raise TypeError("layers must be a list of positive integers")
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
        layer_sizes = [nx] + layers
        for l in range(1, self.__L + 1):
            # layer_size = layers[l - 1]
            # prev_layer_size = nx if l == 1 else layers[l - 2]
            self.__weights[f'W{l}'] = np.random.randn(
                layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2. / layer_sizes[l-1])
            self.__weights[f'b{l}'] = np.zeros((layer_sizes[l], 1))

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
            W = self.__weights[f'W{l}']
            b = self.__weights[f'b{l}']
            A_prev = self.__cache[f'A{l-1}']

            Z = np.dot(W, A_prev) + b
            self.__cache[f'A{l}'] =  1 / (1 + np.exp(-Z))

        return self.__cache[f'A{self.__L}'], self.__cache
