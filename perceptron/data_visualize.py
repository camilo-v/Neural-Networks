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

filePathForInputFile = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/data/letters/data-training.txt"

with open( filePathForInputFile, 'r' ) as INFILE:

    reader = csv.reader( INFILE, delimiter='\t' )

    try:
        for row_line in reader:

            newRowArray = []

            for (index, valueOfCell) in enumerate( row_line ):

                if index == 0:
                    continue
                else:
                    newRowArray.append( valueOfCell )

    except csv.Error as e:
        sys.exit( "File %s, line %d: %s" % (filePathForInputFile, reader.line_num, e) )
