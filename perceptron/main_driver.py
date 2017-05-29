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
import time
import perceptron

#   Custom Modules
sys.path.append(os.path.join(os.path.dirname(__file__), ''))
import utility_functions as uf

# ------------------------------------------------------ Main ---------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" + " Perceptron" + "" )

filePathForTraining_A = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/data/letters/data-A-training.txt"

# ---------------------------------------------- Training Data Loading ------------------------------------------------

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
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Perceptron" + "" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Creating Data Structures..." + "" )

#   Create a Pandas Dataframe from the regular array
#   .T transposes the array, and as_matrix() converts it to a numpy.ndarray
df_y = pd.DataFrame(trainingTargetValues_y).as_matrix()
print(df_y)

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

#   Create a Pandas Dataframe from the regular feature matrix
#   as_matrix() converts it to a numpy.ndarray
df_X = pd.DataFrame(trainingFeatureMatrix_X).as_matrix()
print(df_X)

ppn = perceptron.Perceptron( eta=0.1, n_iter=10 )
ppn.fit( df_X, df_y )

# -------------------------------------------- Perceptron Training Errors ---------------------------------------------
#
#   Plot the misclassification errors versus the number of epochs to see if the Perceptron converges after a given
#   number of epochs.
#
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.tight_layout()
plt.show()
