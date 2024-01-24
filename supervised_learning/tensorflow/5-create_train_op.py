#!/usr/bin/env python3

"""This module contains a function that
creates the training operation for the network
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    loss - loss of the network's prediction
    alpha - learning rate
    returns an operation that trains the network
    using gradient descent
    """
