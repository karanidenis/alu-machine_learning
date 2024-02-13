#!/usr/bin/env python3
import numpy as np

"""
This module conatains a function that
calculates the sensitivity for
each class in a confusion matrix"""


def sensitivity(confusion):
    """calculate sensitivity for each
    class in confusion matrix
    confusion - (classes, classes) confusion matrix
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros(classes)
    for i in range(classes):
        sensitivity[i] = confusion[i, i] / np.sum(confusion[i])
    return sensitivity
