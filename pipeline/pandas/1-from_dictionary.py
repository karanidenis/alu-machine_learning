#!/usr/bin/env python3

"""
This module contains a function that creates
a pandas DataFrame from a dictionary"""

import pandas as pd

mydict = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}
rows = ['A', 'B', 'C', 'D']

df = pd.DataFrame(mydict, index=rows)
