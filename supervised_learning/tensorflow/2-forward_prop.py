#!/usr/bin/env python3

"""This module contains a function that creates
forward propagation graph for the neural network
"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x - placeholder for input data
    layer_sizes - list containing no. of nodes in each layer
    activations - list containing activation functions for each layer
    returns prediction of the network in tensor form
    """
