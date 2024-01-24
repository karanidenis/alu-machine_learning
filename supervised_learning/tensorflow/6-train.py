#!/usr/bin/env python3

"""This module contains a function that
builds, trains, and saves a neural network classifier
"""

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    x_train - contains training input data
    y_train - contains training labels
    x_valid - contains validation input data
    y_valid - contains validation labels
    layer_sizes - list with no. of nodes in each layer
    activations - list with activation functions for each layer
    alpha - learning rate
    iterations - no. of iterations to train over
    save_path - designates where to save the model
    graphs collection: - placeholders x & y
                       - tensors(y_pred, loss, accuracy)
                       - train_op
    returns the path where the model was saved
    """
