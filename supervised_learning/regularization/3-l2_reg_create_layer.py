#!/usr/bin/env python3

"""this module has a function that creates
a tensorflow layer that includes L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    creates a tensorflow layer with L2
    prev -- tensor with output of previous layer
    n - no. of nodes new layer should have
    activation - activatiion function to be used
    lambtha - L2 regularization param
    returns output of the new layer
    """

    # outputs = activation(inputs * kernel + bias)
    # activation - activation function (if not None)
    # kernel is a weights matrix created by the layer,
    # and bias is a bias vector created by the layer
    # (only if use_bias is True).

    # from keras import regularizers
    # from keras.layers import Dense
    # from keras.models import Sequential

    # model = Sequential([
    #     Dense(output_dim=n, input_dim=prev, activation=activation,
    #           kernel_regularizer=lambtha)])
    # return model

    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel,
                            kernel_regularizer=l2)
    return layer(prev)
