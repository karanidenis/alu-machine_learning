#!/usr/bin/env python3

"""
This module contains a function that creates
a pandas DataFrame from a numpy array."""

import pandas as pd


def from_numpy(array):
    """create a pandas dataframe
    columns should be labeled alphabetically"""
    df = pd.DataFrame(array)
    df.columns = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:array.shape[1]]
    return df
