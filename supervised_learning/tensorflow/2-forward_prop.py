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
    a = create_layer(x, layer_sizes[0], activations[0])
    if len(layer_sizes) == len(activations):
        for i in layer_sizes:
            a = create_layer(a, layer_sizes[i], activations[i])

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return sess.run(a)
