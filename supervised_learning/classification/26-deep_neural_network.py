#!/usr/bin/env python3
"""This module is of a deep neural network
perfomaing binary classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Import for object serialization
import os


class DeepNeuralNetwork:
    """Deep Neural Network"""

#     def __init__(self, nx, layers):
#         """class constructor
#         nx - no. of input features
#         layers - list of nodes in each layer of the network"""

#         if not isinstance(nx, int):
#             raise TypeError("nx must be an integer")
#         if nx < 1:
#             raise ValueError("nx must be a positive integer")

#         if (
#             type(layers) is not list
#             or len(layers) < 1
#             or min(layers) < 1
#         ):
#             raise TypeError("layers must be a list of positive integers")

#         # self.L - no. of layers in the neural network
#         self.__L = len(layers)

#         # self.cache - dict of all intermediary values of the network
#         self.__cache = {}

#         # weights - dict of all weights and biases of the network
#         self.__weights = {}
#         for layer in range(self.__L):
#             if layer == 0:
#                 self.__weights['W1'] = np.random.randn(
#                     layers[0], nx) * np.sqrt(2 / nx)
#                 self.__weights['b1'] = np.zeros([layers[0], 1])

#             else:
#                 self.__weights['W{}'.format(layer+1)] = np.random.randn(
#                     layers[layer],
#                     layers[layer-1]) * np.sqrt(2. / layers[layer-1])

#                 self.__weights['b{}'.format(
#                     layer+1)] = np.zeros((layers[layer], 1))

#     @property
#     def L(self):
#         return self.__L

#     @property
#     def cache(self):
#         return self.__cache

#     @property
#     def weights(self):
#         return self.__weights

#     def forward_prop(self, X):
#         """calculate forward propagation of neural network
#         X - contains input data. shape (nx, m)
#         m - no. of examples"""
#         # A = max(0, X)
#         self.__cache['A0'] = X

#         for l in range(1, self.__L+1):
#             W = self.__weights['W{}'.format(l)]
#             b = self.__weights['b{}'.format(l)]
#             A_prev = self.__cache['A{}'.format(l-1)]

#             Z = np.dot(W, A_prev) + b
#             self.__cache['A{}'.format(l)] = 1 / (1 + np.exp(-Z))

#         return self.__cache['A{}'.format(self.__L)], self.__cache

#     def cost(self, Y, A):
#         """calculate cost of model using logistic regression
#         Y - has correct labels for input data
#         A - has activated output of the neuron"""
#         m = Y.shape[1]
#         cost = -(1/m) * np.sum([Y * np.log(A) +
#                                 (1 - Y) * np.log(1.0000001 - A)])
#         return cost

#     def evaluate(self, X, Y):
#         """evaluate the neural network's prediction"""
#         A, _ = self.forward_prop(X)
#         # print(A.shape)
#         cost = self.cost(Y, A)
#         predictions = (A >= 0.5).astype(int)

#         return predictions, cost

#     def gradient_descent(self, Y, cache, alpha=0.05):
#         """calculate one pass of gradient descent on neural network
#         alpha - learning rate"""
#         m = Y.shape[1]
#         A = self.__cache['A{}'.format(self.__L)]
#         dz = A - Y

#         for l in reversed(range(1, self.__L + 1)):
#             # dW = (1 / m) * np.dot(dz, A_prev.T)
#             dw = np.matmul(cache["A{}".format(l - 1)], dz.T) / m
#             db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

#             # Update weights
#             da = cache["A{}".format(l - 1)] * (1 - cache["A{}".format(l - 1)])
#             dz = np.matmul(self.__weights["W{}".format(l)].T, dz) * da
#             self.__weights["W{}".format(l)] -= alpha * dw.T
#             self.__weights["b{}".format(l)] -= alpha * db

#         return self.__weights

#     # def train(self, X, Y, iterations=5000, alpha=0.05):
#     #     """Training the deep neural network"""
#     #     if not isinstance(iterations, int):
#     #         raise TypeError("iterations must be an integer")
#     #     if iterations <= 0:
#     #         raise ValueError("iterations must be a positive integer")
#     #     if not isinstance(alpha, float):
#     #         raise TypeError("alpha must be a float")
#     #     if alpha <= 0:
#     #         raise ValueError("alpha must be positive")

#     #     for i in range(iterations):
#     #         A, cache = self.forward_prop(X)
#     #         self.gradient_descent(Y, cache, alpha)

#     #     return self.evaluate(X, Y)

#     def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
#         """Train the deep neural network"""
#         if not isinstance(iterations, int):
#             raise TypeError("iterations must be an integer")
#         if iterations <= 0:
#             raise ValueError("iterations must be a positive integer")
#         if not isinstance(alpha, float):
#             raise TypeError("alpha must be a float")
#         if alpha <= 0:
#             raise ValueError("alpha must be positive")
#         if not isinstance(step, int):
#             raise TypeError("step must be an integer")
#         if step <= 0 or step > iterations:
#             raise ValueError("step must be positive and <= iterations")

#         costs = []

#         for i in range(iterations):
#             A, cache = self.forward_prop(X)
#             self.gradient_descent(Y, cache, alpha)

#             if i % step == 0 or i == iterations - 1:
#                 cost = self.cost(Y, A)
#                 costs.append(cost)
#                 if verbose:
#                     print("Cost after {} iterations: {}".format(i, cost))

#         if graph:
#             plt.plot(range(0, iterations + 1, step), costs)
#             plt.xlabel('iteration')
#             plt.ylabel('cost')
#             plt.title('Training Cost')
#             plt.show()

#         return self.evaluate(X, Y)

#     def save(self, filename):
#         """saves the instance object to a file in pickle format
#         filename - file which object should be saved"""
#         if not filename.endswith('.pkl'):
#             filename += '.pkl'

#         with open(filename, 'wb') as file:
#             pickle.dump(self, file)

#     @staticmethod
#     def load(filename):
#         """loads a pickled DeepNeuralNetwork object
#         filename - file from which the object should be loaded
#         """
#         if not os.path.exists(filename):
#             return None

#         with open(filename, 'rb') as file:
#             return pickle.load(file)

    def __init__(self, nx, layers):
        """ initialize deep neural network object"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        for x in range(self.L):
            if layers[x] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if x == 0:
                self.__weights = {"W1": np.random.randn(layers[0],
                                  nx) * np.sqrt(2 / nx),
                                  "b1": np.zeros((layers[0], 1))}
            else:
                W = "W" + str(x + 1)
                B = "b" + str(x + 1)
                self.__weights[W] = np.random.randn(
                    layers[x],
                    layers[x - 1]) * np.sqrt(2 / layers[x - 1])
                self.__weights[B] = np.zeros((layers[x], 1))

    @property
    def L(self):
        """ return private w"""
        return self.__L

    @property
    def cache(self):
        """ return private b"""
        return self.__cache

    @property
    def weights(self):
        """ return private a"""
        return self.__weights

    def forward_prop(self, X):
        """ forward prop for deep neural network"""
        self.__cache["A0"] = X
        for x in range(self.__L):
            n = str(x + 1)
            Z = np.matmul(
                self.__weights["W" + n],
                self.__cache["A" + str(x)]) + self.__weights["b" + n]
            self.__cache["A" + n] = 1/(1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ return the cost """
        cost = -(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A))).mean()
        return cost

    def evaluate(self, X, Y):
        """ evaluate to binary 1 or 0"""
        self.forward_prop(X)
        return np.round(self.__cache["A" + str(self.__L)]).astype(
            int), self.cost(Y, self.__cache["A" + str(self.__L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ gradient descent for deep neural network"""
        m = Y.shape[1]
        for x in reversed(range(1, self.__L + 1)):
            AN1 = self.__cache["A" + str(x - 1)]
            A0 = self.__cache["A" + str(x)]
            W0 = self.__weights["W" + str(x)]
            if x == self.__L:
                dz = A0 - Y
            else:
                dz = da * (A0 * (1 - A0))
            db = dz.mean(axis=1, keepdims=True)
            dw = np.matmul(dz, AN1.T) / m
            da = np.matmul(W0.T, dz)
            self.__weights['W' + str(x)] -= (alpha * dw)
            self.__weights['b' + str(x)] -= (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ train deep neural network"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        step_array = list(range(0, iterations + 1, step))
        cost_array = []
        for i in range(iterations + 1):
            if verbose and i in step_array:
                Y_hat, cost = self.evaluate(X, Y)
                cost_array.append(cost)
                print("Cost after {} iterations: {}".format(i, cost))
            if i != iterations:
                self.gradient_descent(Y, *self.forward_prop(X)[0], alpha)
        if graph:
            plt.plot(step_array, cost_array, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ save neural network"""
        if type(filename) is not str:
            return
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except FileNotFoundError:
            return None
