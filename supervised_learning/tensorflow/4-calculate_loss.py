#!/usr/bin/env python3

"""This module contains a function that calculates
the softmax cross-entropy loss
of a prediction
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    y - placeholder for labels of input data
    y_pred - a tensor containing the network's prediction
    returns a tensor containing the loss  of the preiction
    """
    # Convert y_pred to discrete labels
    true_labels = tf.argmax(y, 1)

    loss = tf.losses.mean_squared_error(labels=true_labels, predictions=y_pred)

    return loss
