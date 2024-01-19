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
        layer_sizes = [nx] + layers
        for l in range(1, self.L + 1):
            # layer_size = layers[l - 1]
            # prev_layer_size = nx if l == 1 else layers[l - 2]
            self.weights[f'W{l}'] = np.random.randn(
                layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2. / layer_sizes[l-1])
            self.weights[f'b{l}'] = np.zeros((layer_sizes[l], 1))
