#!/usr/bin/env python3
"""
This module conatains a function that
calculates the f1 score for
each class in a confusion matrix"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    confusion - (classes, classes) confusion matrix
    """
    # precisions = precision(confusion)
    # sensitivitys = sensitivity(confusion)
    # f1 = 2(precision * sensitivity) / (precision + sensitivity)
    # return f1
    classes = confusion.shape[0]
    f1 = np.zeros(classes)
    precisions = precision(confusion)
    sensitivitys = sensitivity(confusion)
    for i in range(classes):
        f1[i] = 2 * (precisions[i] * sensitivitys[i]) / \
            (precisions[i] + sensitivitys[i])
    return f1
