#!/usr/local/bin/python3

# ---------------------------------------------------------------------------------------------------------------------
#
#                                        Bioinformatics Research Group
# 									        http://biorg.cis.fiu.edu/
#                             			Florida International University
#
#   This software is a "Camilo Valdes Work" under the terms of the United States Copyright Act.
#   Please cite the author(s) in any work or product based on this material.
#
#   OBJECTIVE:
#   The purpose of this program is to implement the Perceptron classifier.
#
#   NOTES:
#   Please see the dependencies section below for the required libraries (if any).
#
#   DEPENDENCIES:
#
#       • somePythonModule (http://url_for_module.org)
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

class Perceptron(object):
    """
    Perceptron Classifier.
    This module implements the Perceptron Learning algorithm.

    Parameters:
        eta (float): Learning rate (between 0.0 and 1.0)
        n_iter (int): Number of passes over the entire dataset (epochs).

    Attributes:
        w_ (1-d array): and the weights after fitting.
        errors_ (list): Number of misclassifications in every epoch.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fits the training data, and allows the Perceptron algorithm to learn.
        Args:
            X (array-like): Training vectors, where n_samples is the number of samples and n_features is the number of
            features. Shape = [n_samples, n_feature].
            y (array-like): Target Values. Shape =[n_samples]

        Returns:
            self (object): Returns itself with updated weights.
        """

        #   Initialize weights to zero. Note shape[1] is number of columns, and shape[0] is number of rows.
        self.w_ = np.zeros(1 + X.shape[1])
        print("Initial Weights: ")
        print(self.w_)

        #   Track the misclassifications for a given epoch
        self.errors_ = []

        for _ in range(self.n_iter):

            errors = 0

            #   The 'zip()' function returns a list of tuples, where the i-th tuple contains the i-th element from
            #   each of the argument sequences or iterables.
            for xi, target in zip(X, y):

                #   Perceptron Learning Rule. Recall that self.eta is the learning rate.
                update = self.eta * ( target - self.predict( xi ) )

                # print("-----------")
                # print(update)
                # print(xi)
                # print(xi.shape)

                #   Update the weights (including the bias)
                self.w_[1:] += update * xi
                self.w_[0] += update

                #   Keep track of the errors
                errors += int(update != 0.0)

            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """
        Calculates the Net Input for a neuron.
        Args:
            X (array-like): Training vectors, where n_samples is the number of samples and n_features is the number of
            features. Shape = [n_samples, n_feature].

        Returns:
            float: The net input (dot product) calculated from the input layer.
        """

        #   Return the dot-product of w (transposed) and x
        #   Note: self.w_[0] is basically the "threshold" or so-called "bias unit."
        # print("Bias: " + str(self.w_[0]))
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Returns the class label after a unit step.
        Args:
            X (array-like): Training vectors, where n_samples is the number of samples and n_features is the number of
            features. Shape = [n_samples, n_feature].

        Returns:
            ndarray: A Numpy array value with the expected (predicted) label of the pattern.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
