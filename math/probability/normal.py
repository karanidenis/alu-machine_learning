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
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.mean = float(sum(data) / len(data))
                squared_diff = [(x - self.mean) ** 2 for x in data]
                self.stddev = float(
                    (sum(squared_diff) / (len(data) - 1)) ** 0.5)

    def z_score(self, x):
        """
        calculates z-score of a given x-value
        x is the value
        """
        # score = float((x - self.mean) / self.stddev)
        # return score
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        calculates x-value of the given z-score
        z - z-score
        """
        return (z * self.stddev) + self.mean
        # x_value = (z * self.stddev) + self.mean
        # return x_value

    def pdf(self, x):
        """
        calculates the value of the PDF for a given x-value
        x - the x-value
        """
        pi = 3.1415926536
        e = 2.7182818285
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        pdf = (1 / (self.stddev * (2 * pi) ** 0.5)) * e ** (exponent)
        return pdf

        # pi = 3.1415926536
        # e = 2.7182818285
        # pdf = (1 / (self.stddev * math.sqrt(2 * pi))) * e ** (
        #     -((x - self.mean) ** 2) / (2 * self.stddev ** 2))
        # return pdf
