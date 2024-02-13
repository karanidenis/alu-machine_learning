#!/usr/bin/env python3
import numpy as np

"""
This module conatains a function that
creates a confusion matrix"""


def create_confusion_matrix(labels, logits):
    """
    labels - (m, classes) has correct labels for each data point
    logits - (m, classes) has labels for predicted labels
    """
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    for i in range(m):
        confusion[np.argmax(labels[i]), np.argmax(logits[i])] += 1
    return confusion
