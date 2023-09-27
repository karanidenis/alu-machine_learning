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
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            squared_diff = [(x - self.mean) ** 2 for x in data]
            self.stddev = (sum(squared_diff) / (len(data) - 1)) ** 0.5

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

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value
        x - the x-value
        """
        pi = 3.1415926536
        e = 2.7182818285
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        erf = (2 / (pi ** 0.5)) * (x - ((x ** 3) / 3) + ((x ** 5) / 10) -
                                   ((x ** 7) / 42) + ((x ** 9) / 216))
        print(erf)
        cdf = 0.5 * (1 + erf * (e ** exponent))
        return cdf

        # pi = 3.1415926536
        # e = 2.7182818285
        # cdf = (1 / 2) * (1 + math.erf((x - self.mean) / (self.stddev * 2 ** 0.5)))
        # return cdf
