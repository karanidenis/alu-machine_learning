#!/usr/bin/env python3

"""This module contains a function that
calculates accuracy of a prediction
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y - placeholder for labels of input data
    y_pred - tensor containing network's prediction
    returns a tensor containing the decimal accuracy of
    the prediction
    accuracy = correct_predictions / all_predictions"""
