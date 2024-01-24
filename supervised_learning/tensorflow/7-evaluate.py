#!/usr/bin/env python3

"""This module contains a function that
evaluates the output of a neural network
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    x - contains input data to evaluate
    y -  contains one-hot labels for x
    save_path - location to load the model from
    returns the network's prediction, accuracy, and loss
    """
