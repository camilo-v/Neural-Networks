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
#   The purpose of this program is to run the main code for Project 3, "Bidirectional Associative Memory".
#
#   NOTES:
#   Please see the dependencies section below for the required libraries (if any).
#
#   DEPENDENCIES:
#
#       • Matplotlib.
#       • Pandas
#       • Numpy
#       • Scipy
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
import pprint
import time
import numpy as np

#   Import the main implementation of the Bipolar Associative Memory.
import bipolar_bam

# ------------------------------------------------------ Main ---------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] " +
       "Project 3, Bidirectional Associative Memory" + "" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

pp = pprint.PrettyPrinter( indent=4 )

# -------------------------------------------------- Data Patterns ----------------------------------------------------
#
#   The "dataInBinary" dataset is used for debugging purposes, and the helper function '__l_make_bipolar' handles
#   the conversion to bipolar encoding internally.  The data structures are 2D matrices that store the X-vector and
#   y-vector representations of each letter.
#
#   Indices are: data[X-vector][y-vector]
#
dataInBinary = [
    [[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], [0, 0, 0]], # A
    [[1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0], [0, 0, 1]], # B
    [[0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0]], # C
    [[1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0], [0, 1, 1]], # D
    [[1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1], [1, 0, 0]], # E
    [[1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0], [1, 0, 1]], # F
    [[0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1], [1, 1, 0]], # G
    [[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], [1, 1, 1]]  # H
]

dataInBiPolar = [
    [[-1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1], [-1, -1, -1]],     # A
    [[1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1], [-1, -1, 1]],      # B
    [[-1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1], [-1, 1, -1]],   # C
    [[1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1], [-1, 1, 1]],       # D
    [[1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, 1], [1, -1, -1]],      # E
    [[1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1], [1, -1, 1]],     # F
    [[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1], [1, 1, -1]],      # G
    [[1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1], [1, 1, 1]]          # H
]

tempData = []

for XvectorIndex in range(0, 8):
    tempData.append([dataInBiPolar[XvectorIndex][0], dataInBiPolar[XvectorIndex][1]])

print("*************************************")
pp.pprint( tempData )


# --------------------------------------------------- BAM Training ----------------------------------------------------
#
#   Create a BAM object and pass it the pattern dataset that we wish to load into the memory.
#
b = bipolar_bam.BAM( dataInBiPolar, isBipolar=True )


print ("\n")
print ( "Matrix: " )
pp.pprint(b.get_bam_matrix())

print ('\n')

# ---------------------------------------------------- BAM Testing ----------------------------------------------------
#
#   Once we have loaded the datasets into the BAM, we can go ahead and test it.

#   Pattern A
print ("[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1] ---> ",
       b.get_assoc([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1]))


print ('\n\n')

# -------------------------------------------------- End --------------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] Done." )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
