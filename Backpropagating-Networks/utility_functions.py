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
#   The purpose of this program is to define reusable utility functions for Project 2.  Namely, this module contains
#   the loading functions for training datasets (with labels), and the loading functions for testing datasets.
#
#   NOTES:
#   Please see the dependencies section below for the required libraries (if any).
#
#   DEPENDENCIES:
#
#       • Standard python libraries.
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
import sys
import csv

# --------------------------------------------------- Functions -------------------------------------------------------

def readTrainingData(filePath):
    """
    Utility function for reading in the training data.
    Args:
        filePath (string): The path of the file that contains the training data.

    Returns:
        Tuple with target values (y) and feature matrix (X).
    """

    filePathForInputFile = filePath

    #   Array with target values
    targetValues_y = []

    #   Create a feature matrix
    w, h = 20, 25
    featureMatrix_X = [[0 for x in range( w )] for y in range( h )]

    patternNumber = 0

    with open( filePathForInputFile, 'r' ) as INFILE:

        reader = csv.reader( INFILE, delimiter='\t' )

        try:
            for row_line in reader:  # row_line is a list, not a string

                if ''.join( row_line ).startswith( "#" ):
                    # print( "\nComment. Starting new letter recognition..." )
                    continue
                else:
                    patternArray = []
                    patternLabel = ""

                    for (index, valueOfCell) in enumerate( row_line ):

                        if index != (len( row_line ) - 1):
                            patternArray.append( int( valueOfCell ) )
                            featureMatrix_X[patternNumber][index] = int( valueOfCell )

                        if index == (len( row_line ) - 1):
                            patternLabel = int( valueOfCell )

                    # print( patternArray )
                    # print( patternLabel )

                    targetValues_y.append( patternLabel )

                    patternNumber += 1

        except csv.Error as e:
            sys.exit( "File %s, line %d: %s" % (filePathForInputFile, reader.line_num, e) )

    return targetValues_y, featureMatrix_X

def readTestingData(filePath):
    """
    Utility function for reading in the testing data (does not have a pattern label at the end of the line).
    Args:
        filePath (string): The path of the file that contains the testing data.

    Returns:
        array A two-dimensional array (feature matrix) with the data for the given testing set.
    """

    filePathForInputFile = filePath

    #   Array with target values for testing set.
    targetValues_y = []

    #   Create a feature matrix
    w, h = 20, 25
    featureMatrix_X = [[0 for x in range( w )] for y in range( h )]

    patternNumber = 0

    with open( filePathForInputFile, 'r' ) as INFILE:

        reader = csv.reader( INFILE, delimiter='\t' )

        try:
            for row_line in reader:  # row_line is a list, not a string

                if ''.join( row_line ).startswith( "#" ):
                    # print( "\nComment. Starting new letter recognition..." )
                    continue
                else:
                    patternArray = []
                    patternLabel = ""

                    for (index, valueOfCell) in enumerate( row_line ):

                        if index != (len( row_line ) - 1):
                            patternArray.append( int( valueOfCell ) )
                            featureMatrix_X[patternNumber][index] = int( valueOfCell )

                        if index == (len( row_line ) - 1):
                            patternLabel = int( valueOfCell )

                    # print( patternArray )
                    # print( patternLabel )

                    targetValues_y.append( patternLabel )

                    patternNumber += 1

        except csv.Error as e:
            sys.exit( "File %s, line %d: %s" % (filePathForInputFile, reader.line_num, e) )

    return targetValues_y, featureMatrix_X
