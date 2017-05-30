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
#   The purpose of this program is to run the main code for Project 1.
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

#   Python Modules
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

#   Custom Modules
sys.path.append(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(os.path.join(os.path.dirname(__file__), 'perceptron'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'adaline'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'hebbian'))
import utility_functions as uf
import perceptron
import adaline
import hebbian

# ------------------------------------------------------ Main ---------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" + " Perceptron" + "" )

#   Training Data for letter A
filePathForTraining_A = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/data/letters/data-A-training.txt"

#   Paths for testing sets
pathForTestingSet_1 = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/data/letters/data-testing_set_1.txt"
pathForTestingSet_2 = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/data/letters/data-testing_set_2.txt"
pathForTestingSet_3 = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/data/letters/data-testing_set_3.txt"

# ---------------------------------------------- Training Data Loading ------------------------------------------------

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Loading Training Data..." + "" )

#   Load the training data for the letter A, and get back an array with target values (y) and feature matrix (X).
trainingTargetValues_y, trainingFeatureMatrix_X = uf.readTrainingData(filePathForTraining_A)

# -------------------------------------------- Training Data Inspection -----------------------------------------------

print("Target Values (y): ")
print(trainingTargetValues_y)

print("Feature Matrix (X): ")
print(trainingFeatureMatrix_X)

# --------------------------------------------------- Perceptron ------------------------------------------------------
#
#   Train the perceptron algorithm.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Creating Data Structures..." + "" )

#   Create a Pandas Dataframe from the regular array
#   .T transposes the array, and as_matrix() converts it to a numpy.ndarray
df_y = pd.DataFrame(trainingTargetValues_y).as_matrix()
# print(df_y)

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

#   Create a Pandas Dataframe from the regular feature matrix
#   as_matrix() converts it to a numpy.ndarray
df_X = pd.DataFrame(trainingFeatureMatrix_X).as_matrix()
# print(df_X)

#
#   Initialize the Perceptron, and train it with the "fit" method.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Initializing Perceptron..." + "" )
ppn = perceptron.Perceptron( eta=0.1, n_iter=10 )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Training Perceptron..." + "" )
ppn.fit( df_X, df_y )

# -------------------------------------------- Perceptron Training Errors ---------------------------------------------
#
#   Plot the misclassification errors versus the number of epochs to see if the Perceptron converges after a given
#   number of epochs.
#

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Plotting Training Errors..." + "" )

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.tight_layout()
# plt.show()

# ----------------------------------------------- Testing Data Loading ------------------------------------------------
#
#   Now that we have a fully trainined Perceptron, we can go ahead and load the testing data to "test" it.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Loading Testing data..." + "" )

#   Read-in the data for the testing sets
testingSet1FeatureMatrix_X = uf.readTestingData( pathForTestingSet_1 )
testingSet2FeatureMatrix_X = uf.readTestingData( pathForTestingSet_2 )
testingSet3FeatureMatrix_X = uf.readTestingData( pathForTestingSet_3 )

#   Data structures for testing sets (feature matrices only)
df_testingSet_1_X = pd.DataFrame( testingSet1FeatureMatrix_X ).as_matrix()
df_testingSet_2_X = pd.DataFrame( testingSet2FeatureMatrix_X ).as_matrix()
df_testingSet_3_X = pd.DataFrame( testingSet3FeatureMatrix_X ).as_matrix()

# ------------------------------------------------ Perceptron Testing -------------------------------------------------
#
#   Now that we have loaded the testing sets as feature matrices, we will use the Perceptron's class "predict" method
#   to test the classifier with a given testing set
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Running testing datasets..." + "" )

testingSet_1_error = 0
testingSet_2_error = 0
testingSet_3_error = 0

#   Testing Dataset I
for xi, target in zip(df_testingSet_1_X, df_y):
    testResult = target - ppn.predict( xi )
    testingSet_1_error += int( testResult != 0.0 )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Test Set 1 Error: " +
       str(testingSet_1_error) )

#   Testing Dataset II
for xi, target in zip(df_testingSet_2_X, df_y):
    testResult = target - ppn.predict( xi )
    testingSet_2_error += int( testResult != 0.0 )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Test Set 2 Error: " +
       str(testingSet_2_error) )

#   Testing Dataset III
for xi, target in zip(df_testingSet_3_X, df_y):
    testResult = target - ppn.predict( xi )
    testingSet_3_error += int( testResult != 0.0 )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Test Set 3 Error: " +
       str(testingSet_3_error) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )


# ----------------------------------------------------- Adaline -------------------------------------------------------
#
#   Train the Adaline classifier algorithm.
#

#   Matplotlib objects to hold dual-pane figure
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = adaline.Adaline(n_iter=10, eta=0.01)
ada1.fit( df_X, df_y )
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-Squared-Error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = adaline.Adaline(n_iter=10, eta=0.0001).fit( df_X, df_y )
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-Squared-Error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.tight_layout()
plt.show()
