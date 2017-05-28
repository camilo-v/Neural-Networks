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
#   The purpose of this program is to visualize the input data, simple 4x5 letter representations, as a two-class
#   problem.
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
import csv

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
                        patternArray.append( valueOfCell )
                        featureMatrix_X[patternNumber][index] = valueOfCell

                    if index == (len( row_line ) - 1):
                        patternLabel = valueOfCell

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
