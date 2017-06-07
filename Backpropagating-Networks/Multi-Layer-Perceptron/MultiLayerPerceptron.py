#!/usr/local/bin/python3

# ---------------------------------------------------------------------------------------------------------------------
#
#                                        Bioinformatics Research Group
#                                           http://biorg.cis.fiu.edu/
#                                       Florida International University
#
#   This software is a "Camilo Valdes Work" under the terms of the United States Copyright Act.
#   Please cite the author(s) in any work or product based on this material.  MLP code is based on the code of
#   Sebastian Raschka at https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch12/ch12.ipynb.
#
#   OBJECTIVE:
#   The purpose of this program is implement the basic Multi-Layered Perceptron algorithm.
#
#   NOTES:
#   Please see the dependencies section below for the required libraries (if any).
#
#   DEPENDENCIES:
#
#       • Matplotlib.
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

#   Python Modules
import os
import sys
import time
import numpy as np
from scipy.special import expit

class MultiLayerPerceptron( object ):
    """
    Multi-Layered Perceptron Classifier.
    This module implements a basic feed-forward neural network as a Multi-Layered Perceptron classifier.

    Parameters:
        n_output (int): Number of output units (PEs), and the same as the number of unique class labels.
        n_features (int): Number of features (columns, i.e., dimmensions) in the training dataset.
        n_hidden (int): Number of hidden units (PEs).
        l1 (float): Lambda value for L1-regularization. Set to 0.0 to disable.
        l2 (float): Lambda value for L2-regularization. Set to 0.0 to disable.
        epochs (int): Number of epochs to run through the training dataset.
        eta (float): Learning Rate.
        alpha (float): Momemtum Constant.  This is the multiplicative factor for gradient descent.
        decrease_const (float): Shrinks the learning rate after each epoch.
        shuffle (bool): If set to true, will shuffle the training dataset at each epoch to prevent cycles.
        minibatches (int): Divides training data into k minibranches for computational efficiency.
        random_state (int): Random initialization for shuffling and setting the weights

    Attributes:
        cost_ (list): Sum of Squared Errors (SSE) for a given epoch.
    """

    def __init__( self, n_output, n_features, n_hidden=30,
                  l1=0.0, l2=0.0, epochs=500, eta=0.001,
                  alpha=0.0, decrease_const=0.0, shuffle=True,
                  minibatches=1, random_state=None ):

        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches


    def _encode_labels( self, y, k):
        """
        Encode the class labels into a "one-hot" representation vector.
        Args:
            y (array): Array of target values with dimensions [n_samples]
            k:

        Returns:
            onehot A tuple with array and shape dimensions (n_labels, n_samples)
        """

        onehot = np.zeros((k, y.shape[0]))

        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0

        return onehot


    def _initialize_weights(self):
        """
        Convenince method for initializing the weights with small random numbers.
        Returns:
            w1, w2
        """

        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden*(self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)

        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output*(self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)

        return w1, w2


    def _sigmoid(self, z):
        """
        Computes a basic sigmoid logistic function
        Args:
            z:

        Returns:
            An SciPy ndarray with a logistic function applied expit(x) = 1/(1+exp(-x)).
        """

        return expit(z)


    def _sigmoid_gradient(self, z):
        """
        Calgulates the gradient of the sigmoid logistic function
        Args:
            z:

        Returns:
            Derivative of sigmoid function.
        """

        sg = self._sigmoid(z)

        return sg * (1.0 - sg)


    def _add_bias_unit(self, X, how='column'):
        """
        Programatically add the bias units (value 1) to the input feature matrix X, starting at index 0.
        Args:
            X:  Input feature matrix.
            how:

        Returns:
            New feature matrix with bias units (column of 1s) added.
        """

        if how == 'column':
            #   Note shape[1] is number of columns, and shape[0] is number of rows.
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError('`how` must be set to `column` or `row`')

        return X_new
