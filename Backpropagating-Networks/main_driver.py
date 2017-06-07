#!/usr/local/bin/python3

# ---------------------------------------------------------------------------------------------------------------------
#
#                                        Bioinformatics Research Group
#                                           http://biorg.cis.fiu.edu/
#                                       Florida International University
#
#   This software is a "Camilo Valdes Work" under the terms of the United States Copyright Act.
#   Please cite the author(s) in any work or product based on this material.
#
#   OBJECTIVE:
#   The purpose of this program is to run the main code for Project 2, "Backpropagation Networks".
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
import matplotlib.pyplot as plt
import pandas as pd
import time

#   Custom Modules
sys.path.append(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Multi-Layer-Perceptron'))

import utility_functions as uf

# ------------------------------------------------------ Main ---------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] " + "Project 2, Backpropagation Networks" + "" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

#   Base directory for project modules, files, etc.
baseDirectoryForProject = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/Backpropagating-Networks"
#   Training Data for letters A and E.

filePathForTraining = baseDirectoryForProject + "/data/letters/data-A-training.txt"
# filePathForTraining = baseDirectoryForProject + "/data/letters/data-E-training.txt"

#   Paths for testing sets
pathForTestingSet_1 = baseDirectoryForProject + "/data/letters/data-testing_set_1.txt"
pathForTestingSet_2 = baseDirectoryForProject + "/data/letters/data-testing_set_2.txt"
pathForTestingSet_3 = baseDirectoryForProject + "/data/letters/data-testing_set_3.txt"

# ---------------------------------------------- Training Data Loading ------------------------------------------------

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Loading Training Data..." + "" )

#   Load the training data for the letter A, and get back an array with target values (y) and feature matrix (X).
trainingTargetValues_y, trainingFeatureMatrix_X = uf.readTrainingData( filePathForTraining )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Creating Data Structures..." + "" )

#   Create a Pandas Dataframe from the regular target array. Note tha as_matrix() converts it to a numpy.ndarray
#   The target values (ground truth)
df_y = pd.DataFrame(trainingTargetValues_y).as_matrix()

#   Create a Pandas Dataframe from the regular feature matrix
df_X = pd.DataFrame(trainingFeatureMatrix_X).as_matrix()

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

# ----------------------------------------------- Testing Data Loading ------------------------------------------------
#
#   Load the test datasets.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Loading Testing data..." + "" )

#   Read-in the data for the testing sets
testingSet1FeatureMatrix_X = uf.readTestingData( pathForTestingSet_1 )
testingSet2FeatureMatrix_X = uf.readTestingData( pathForTestingSet_2 )
testingSet3FeatureMatrix_X = uf.readTestingData( pathForTestingSet_3 )

#   Data structures for testing sets (feature matrices only)
df_testingSet_1_X = pd.DataFrame( testingSet1FeatureMatrix_X ).as_matrix()
df_testingSet_2_X = pd.DataFrame( testingSet2FeatureMatrix_X ).as_matrix()
df_testingSet_3_X = pd.DataFrame( testingSet3FeatureMatrix_X ).as_matrix()

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )


# --------------------------------------------------- Backpropagation -------------------------------------------------
#
#   The following section trains a Multi Layered Perceptron (MLP) with the standard Backpropagation algorithm.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation..." + "" )





print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

# --------------------------------------------- Backpropagation with Momemtum -----------------------------------------
#
#   The following section trains a Multi Layered Perceptron (MLP) with the Backpropagation  plus momemtum algorithm.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation with Momemtum..." + "" )






print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

# ------------------------------------------- Backpropagation with Nguyen-Widrow --------------------------------------
#
#   The following section trains a Multi Layered Perceptron (MLP) with Backpropagation and Nguyen-Widrow initialization
#   algorithm, but with no momemtum.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation with Nguyen-Widrow..." + "" )






print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

# -------------------------------------- Backpropagation with Momemtum and Nguyen-Widrow ------------------------------
#
#   The following section trains a Multi Layered Perceptron (MLP) with Backpropagation and Momemtum, along with the
#   Nguyen-Widrow initialization algorithm.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation with Momemtum and Nguyen-Widrow..." + "" )





# ----------------------------------------------------------- End -----------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] Done." )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
