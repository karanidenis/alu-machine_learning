#!/usr/bin/env python3
"""
This class represents a Binomial distribution
"""


class Binomial:
    """
    This class represents a Binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        function initializes the Binomial distribution
        data - list of the data to be used to estimate the distribution
        n - number of Bernoulli trials
        p - probability of a “success”
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.p = sum(data) / len(data)
            self.n = round(len(data) * self.p)
            self.p = self.n / len(data)
            self.n = int(self.n)
