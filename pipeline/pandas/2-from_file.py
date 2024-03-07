#!/usr/bin/env python3

"""
This module contains a function that
loads data from a file into a pandas DataFrame"""

import pandas as pd


def from_file(filename, delimiter):
    """
    create a pandas DataFrame from a file
    filename: file to load from
    delimiter: the column separator
    """
    df = pd.read_csv(filename, delimiter)
    return df
