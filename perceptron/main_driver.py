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

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import csv
import perceptron

# ------------------------------------------------------ Main ---------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" + " Perceptron" + "" )

filePathForInputFile = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/data/letters/data-A-training.txt"

#   Array with target values
targetValues_y = []

#   Create a feature matrix
w, h = 20, 25
featureMatrix_X = [[0 for x in range(w)] for y in range(h)]

# --------------------------------------------------- File Loading ----------------------------------------------------

patternNumber = 0

with open( filePathForInputFile, 'r' ) as INFILE:

    reader = csv.reader( INFILE, delimiter='\t' )

    try:
        for row_line in reader:     # row_line is a list, not a string

            if ''.join(row_line).startswith("#"):
                print( "\nComment. Starting new letter recognition..." )
            else:
                patternArray = []
                patternLabel = ""

                for (index, valueOfCell) in enumerate( row_line ):

                    if index != (len( row_line ) - 1):
                        patternArray.append( int(valueOfCell) )
                        featureMatrix_X[patternNumber][index] = int(valueOfCell)

                    if index == (len( row_line ) - 1):
                        patternLabel = int(valueOfCell)

                print(patternArray)
                print(patternLabel)

                targetValues_y.append(patternLabel)

                patternNumber += 1

    except csv.Error as e:
        sys.exit( "File %s, line %d: %s" % (filePathForInputFile, reader.line_num, e) )

# ------------------------------------------------- Data Inspection ---------------------------------------------------

print("Target Values (y): ")
print(targetValues_y)

print("Feature Matrix (X): ")
print(featureMatrix_X)

# --------------------------------------------------- Perceptron ------------------------------------------------------

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" + " Perceptron" + "" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" + " Creating Data Structures..." + "" )

#   Create a Pandas Dataframe from the regular array
#   .T transposes the array, and as_matrix() converts it to a numpy.ndarray
df_y = pd.DataFrame(targetValues_y).as_matrix()
print(df_y)

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S',time.localtime()) + " ]" )

#   Create a Pandas Dataframe from the regular feature matrix
#   as_matrix() converts it to a numpy.ndarray
df_X = pd.DataFrame(featureMatrix_X).as_matrix()
print(df_X)

ppn = perceptron.Perceptron( eta=0.1, n_iter=10 )
ppn.fit( df_X, df_y )

# ----------------------------------------------------- Errors --------------------------------------------------------
#
#   Plot the misclassification errors versus the number of epochs to see if the Perceptron converges after a given
#   number of epochs.
#
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.tight_layout()
plt.show()
