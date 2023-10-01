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
        
    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes" k without using external libraries.
        k - The number of "successes" for which you want to calculate the PMF.
        Returns the PMF value for k.
        """
        k = int(k)  # Convert k to an integer
        if k < 0 or k > self.n:
            return 0
        else:
            # Calculate the binomial coefficient (n choose k) using dynamic programming
            def calculate_binomial_coefficient(n, k):
                dp = [[0] * (k + 1) for _ in range(n + 1)]
                for i in range(n + 1):
                    dp[i][0] = 1
                for i in range(1, n + 1):
                    for j in range(1, min(i, k) + 1):
                        dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                return dp[n][k]

            binomial_coeff = calculate_binomial_coefficient(self.n, k)

            # Calculate (self.p ** k) and ((1 - self.p) ** (self.n - k))
            p_power_k = self.p
            q_power_n_minus_k = 1 - self.p
            for i in range(1, k):
                p_power_k *= self.p
                q_power_n_minus_k *= (1 - self.p)

            pmf = binomial_coeff * p_power_k * q_power_n_minus_k
            return pmf
