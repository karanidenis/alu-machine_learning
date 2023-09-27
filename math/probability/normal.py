#!/usr/bin/env python3
"""
This class represents a Normal distribution
"""


class Normal:
    """
    This class represents a Normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        function initializes the normal distribution
        data - list of the data to be used to estimate the distribution
        mean - mean of the distribution
        stddev - standard deviation
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = stddev
            self.mean = mean
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = sum(data) / len(data)
                self.stddev = (sum([(x - self.mean) ** 2 for x in data]))
