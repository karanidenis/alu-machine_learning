#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# take last 10 rows of the columns High and Close
# and convert them to a numpy array
A = df.tail(10)[['High', 'Close']].to_numpy()

print(A)