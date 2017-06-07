#!/usr/local/bin/python3

# ---------------------------------------------------------------------------------------------------------------------
#
#                                         Bioinformatics Research Group
#                                           http://biorg.cis.fiu.edu/
#                                       Florida International University
#
#   This software is a "Camilo Valdes Work" under the terms of the United States Copyright Act. Please cite the
#   author(s) in any work or product based on this material.  Base implementation based on Sebastian Raschka at
#   https://github.com/rasbt/python-machine-learning-book/.
#
#   OBJECTIVE:
#   The purpose of this program is to implement the Adaptive Linear Neuron (ADALINE) classifier.
#
#   NOTES:
#   Please see the dependencies section below for the required libraries (if any).
#
#   DEPENDENCIES:
#
#       • Pandas
#       • Numpy
#
#   The above libraries & modules are required. You can check the modules currently installed in your
#   system by running: python -c "help('modules')"
#
#   USAGE:
#   Run the program with the "--help" flag to see usage instructions.
#
#   AUTHOR:	Camilo Valdes (cvalde03@fiu.edu)
#           Bioinformatics Research Group,
#           School of Computing and Information Sciences,
#           Florida International University (FIU)
#
#
# ---------------------------------------------------------------------------------------------------------------------

# 	Python Modules
import numpy as np

class Adaline(object):
    """
    Perceptron Classifier.
    This module implements the Perceptron Learning algorithm.

    Parameters:
        eta (float): Learning rate (between 0.0 and 1.0)
        n_iter (int): Number of passes over the entire dataset (epochs).

    Attributes:
        w_ (1-d array): Weights after fitting.
        cost_ (list): Sum-of-squares cost function value in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50):
        """
        Default initilizer.
        Args:
            eta: The learning rate.
            n_iter: number of iterations (epochs).
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit the Adaline instance with the training data.
        Args:
            X (numpy.ndarray): Feature matrix, where n_samples is the number of samples and n_features is the
                number of features.
            y (numpy.ndarray): Target values.

        Returns:
            self An object type.
        """

        self.w_ = np.array(np.zeros( 1 + X.shape[1] ))
        self.w_ = self.w_[:, np.newaxis]    # Adds a new axis -> 2D array. Required to update the weights.
        self.cost_ = []

        for i in range(self.n_iter):

            output = self.activation( X )

            errors = (y - output)

            #   Calculate the gradient based on the whole training dataset for weights 1 to m
            #   Note that np.asarray(self.w_[1:]) is required so that Numpy can see the vector of weights
            #   correctly and it can perform the dot product.
            self.w_[1:] = np.add( np.asarray(self.w_[1:]), self.eta * X.T.dot( errors ) )

            #   Calculate the gradient based on the whole training dataset
            self.w_[0] += self.eta * errors.sum()

            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """
        Net input calculation for a given P.E.
        Args:
            X (numpy.ndarray): Feature matrix.

        Returns:
            numpy.ndarray Sum of net inputs
        """

        return np.dot( X, self.w_[1:] ) + self.w_[0]

    def activation(self, X):
        """
        Method for computing the linear activation function.
        Args:
            X (numpy.ndarray): Feature matrix.

        Returns:
            numpy.ndarray Activation as calculated by the net input.
        """

        return self.net_input(X)

    def predict(self, X):
        """
        Estimate the class label for a given pattern
        Args:
            X (numpy.ndarray): Feature matrix.

        Returns:
            ndarray: A Numpy array value with the expected (predicted) label of the pattern.
        """

        return np.where(self.activation(X) >= 0.0, 1, -1)
