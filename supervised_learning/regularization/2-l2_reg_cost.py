#!/usr/bin/env python3

"""This module has a function that calculates
cost of  a neural entwork with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost):
    """calculate cost of the network
    cost - tensor containing cost of network without L2
    returns a tensor containing the cost of the network
    accounting for L2 regularization"""

    # Retrieve regularization losses
    # regularization_losses = tf.add_n(
    # tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Add the regularization losses to the original cost
    # total_cost = cost + regularization_losses
    # return total_cost

    return cost + tf.losses.get_regularization_losses()
