#!/usr/bin/env python3

"""This module contains a function that
returns tensor output of the layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev - tensor output of previous layer
    n - no. of nodes in layer to create
    activation - activation function
    layer - name of layers
    """