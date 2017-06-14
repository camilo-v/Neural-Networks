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
import os
import sys
import pprint
import time

#   Import the main implementation of the Bipolar Associative Memory.
import bipolar_bam

# ------------------------------------------------------ Main ---------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] " +
       "Project 3, Bidirectional Associative Memory" + "" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )


# -------------------------------------------------- Data Patterns ----------------------------------------------------
#
#
data_pairs  = [
        [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0]],
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0]]
        ]

#   Data pattern for letter A.
A_data = [
    [[-1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1], [-1, -1, -1]]
]

#   Data pattern for letter B.
B_data = [
    [[1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1], [-1, -1, 1]]
]

b = bipolar_bam.BAM(A_data)


pp = pprint.PrettyPrinter(indent=4)

print ('Matrix: ')
pp.pprint(b.get_bam_matrix())

print ('\n')
print ('[1, 0, 1, 0, 1, 0] ---> ', b.get_assoc([1, 0, 1, 0, 1, 0]))
print ('[1, 1, 1, 0, 0, 0] ---> ', b.get_assoc([1, 1, 1, 0, 0, 0]))




# -------------------------------------------------- End --------------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] Done." )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
