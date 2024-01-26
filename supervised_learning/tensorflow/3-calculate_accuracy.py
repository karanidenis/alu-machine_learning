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

    # Convert y_pred to discrete labels
    predicted_labels = tf.argmax(y_pred, 1)
    true_labels = tf.argmax(y, 1)

    # Compare predicted labels with true labels
    correct_predictions = tf.equal(predicted_labels, true_labels)

    # calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
