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
        alpha (float): Momentum Constant.  This is the multiplicative factor for gradient descent.
        decrease_const (float): Shrinks the learning rate after each epoch.
        shuffle (bool): If set to true, will shuffle the training dataset at each epoch to prevent cycles.
        minibatches (int): Divides training data into k minibranches for computational efficiency.
        random_state (int): Random initialization for shuffling and setting the weights

    Attributes:
        cost_ (list): Sum of Squared Errors (SSE) for a given epoch.
    """

    def __init__( self, n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0,
                  decrease_const=0.0, shuffle=True, minibatches=1, random_state=None ):

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

    def _feedforward(self, X, w1, w2):
        """
        Calculates a feedforward step for the network.
        Args:
            X (array): Input layer with original features, has shape [n_samples, n_features].
            w1 (array): Weight matrix for input layer to hidden layer [n_hidden_units, n_features].
            w2 (array): Weight matrix for hidden layer to output layer [n_output_units, n_hidden_units].

        Returns:
            a1 (array): Input values with biast unit [n_samples, n_features].
            z2 (array): Net input of hidden layer [n_hidden, n_samples].
            a2 (array): Activation of hidden layer [n_hidden + 1, n_samples].
            z3 (array): Net input of hidden layer [n_output_units, n_samples].
            a3 (array): Activation of output layer [n_output_units, n_samples].
        """

        a1 = self._add_bias_unit(X, how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='row')
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)

        return a1, z2, a2, z3, a3

    def _L2_reg(self, lambda_, w1, w2):
        """
        Computes the L2 Regularization cost.
        Args:
            lambda_:
            w1:
            w2:

        Returns:

        """

        return (lambda_ / 2.0) * (np.sum( w1[:, 1:] ** 2 ) + np.sum( w2[:, 1:] ** 2 ))

    def _L1_reg(self, lambda_, w1, w2):
        """
        Computes the L1 Regularization cost.
        Args:
            lambda_:
            w1:
            w2:

        Returns:

        """

        return (lambda_ / 2.0) * (np.abs( w1[:, 1:] ).sum() + np.abs( w2[:, 1:] ).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        """
        Calculates the cost function.
        Args:
            y_enc:  "One-Hot" encoded class labels for target, t.
            output: Activation of the output layer (feedforward)
            w1: Weight matrix for input layer to hidden layer.
            w2: Weight matrix for hidden layer to output layer.

        Returns:
            cost (float) The regularized cost.
        """

        term1 = -y_enc * (np.log( output ))
        term2 = (1.0 - y_enc) * np.log( 1.0 - output )
        cost = np.sum( term1 - term2 )
        L1_term = self._L1_reg( self.l1, w1, w2 )
        L2_term = self._L2_reg( self.l2, w1, w2 )
        cost = cost + L1_term + L2_term

        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        """
        Computes the gradient step using the backpropagation algorithm.
        Args:
            a1: Input values with bias unit.
            a2: Activation of hidden layer.
            a3: Activation of output layer.
            z2: Net input of hidden layer.
            y_enc: "One-Hot" encoded class labels for target, t.
            w1: Weight matrix for input layer to hidden layer.
            w2: Weight matrix for hidden layer to output layer.

        Returns:
            grad1 (array) Gradient of the weight matrix w1, has shape [n_hidden_units, n_features].
            grad2 (array) Gradient of the weight matrix w2, has shape [n_output_units, n_hidden_units].
        """

        #   Backpropagation
        sigma3 = a3 - y_enc
        z2 = self._add_bias_unit( z2, how='row' )
        sigma2 = w2.T.dot( sigma3 ) * self._sigmoid_gradient( z2 )
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot( a1 )
        grad2 = sigma3.dot( a2.T )

        #   Regularization
        grad1[:, 1:] += self.l2 * w1[:, 1:]
        grad1[:, 1:] += self.l1 * np.sign( w1[:, 1:] )
        grad2[:, 1:] += self.l2 * w2[:, 1:]
        grad2[:, 1:] += self.l1 * np.sign( w2[:, 1:] )

        return grad1, grad2


    def predict(self, X):
        """
        Prediction function for class labels.
        Args:
            X: Input layer with original features, has shape [n_samples, n_features].

        Returns:
            y_pred (array) Predicted class labels, has shape [n_samples].
        """

        if len( X.shape ) != 2:
            raise AttributeError( 'X must be a [n_samples, n_features] array.\n'
                                  'Use X[:,None] for 1-feature classification,'
                                  '\nor X[[i]] for 1-sample classification' )

        a1, z2, a2, z3, a3 = self._feedforward( X, self.w1, self.w2 )

        y_pred = np.argmax( z3, axis=0 )

        return y_pred

    def fit(self, X, y, print_progress=False):
        """
        Training function for learning the weights from the training dataset.
        Args:
            X (array): Input layer with original features, has shape [n_samples, n_features].
            y (array): Target class labels from training data, has shape [n_samples].
            print_progress (bool): Flag for printing the progress through the epochs.

        Returns:
            Fitted values in a self object.
        """

        #   Retains the cost for each epoch so we can plot it later.
        self.cost_ = []

        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels( y, self.n_output )

        delta_w1_prev = np.zeros( self.w1.shape )
        delta_w2_prev = np.zeros( self.w2.shape )

        for i in range( self.epochs ):

            #   Learning rate
            self.eta /= (1 + self.decrease_const * i)

            if print_progress:
                sys.stderr.write( '\nEpoch: %d/%d' % (i + 1, self.epochs) )
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation( y_data.shape[0] )
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split( range( y_data.shape[0] ), self.minibatches )

            for idx in mini:
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward( X_data[idx],
                                                        self.w1,
                                                        self.w2 )
                cost = self._get_cost( y_enc=y_enc[:, idx],
                                       output=a3,
                                       w1=self.w1,
                                       w2=self.w2 )

                self.cost_.append( cost )

                #   Compute gradient via Backpropagation
                grad1, grad2 = self._get_gradient( a1=a1, a2=a2,
                                                   a3=a3, z2=z2,
                                                   y_enc=y_enc[:, idx],
                                                   w1=self.w1,
                                                   w2=self.w2 )

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2

                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))

                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self

