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
#   The purpose of this program is implement a basic Bipolar Associative Memory network.
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


class BAM(object):

    def __init__(self, data):
        """
        Designated Initializer.
        Args:
            data: 2D matrix of data pairs
        """

        self.AB = []

        # store associations in bipolar form to the array
        for item in data:
            self.AB.append(
                [self.__l_make_bipolar(item[0]),
                 self.__l_make_bipolar(item[1])]
                )

        self.len_x = len(self.AB[0][1])
        self.len_y = len(self.AB[0][0])

        # create empty BAM matrix
        self.M = [[0 for x in range(self.len_x)] for x in range(self.len_y)]

        # compute BAM matrix from associations
        self.__create_bam()

    def __create_bam(self):
        """
        Creates a Bidirectional Associative Memory object.
        Returns:

        """

        for assoc_pair in self.AB:
          X = assoc_pair[0]
          Y = assoc_pair[1]
         # calculate M
          for idx, xi in enumerate(X):
            for idy, yi in enumerate(Y):
              self.M[idx][idy] += xi * yi

    def get_assoc(self, A):
        """
        Returns the association for the input vector A.
        Args:
            A: Vector input.

        Returns:

        """

        A = self.__mult_mat_vec(A)
        return self.__threshold(A)

    def get_bam_matrix(self):
        """
        Utility Method for returning a BAM matrix
        Returns:
            self.M A BAM matrix.
        """

        return self.M

    def __mult_mat_vec(self, vec):
        """
        Multiples the input vector with a BAM matrix.
        Args:
            vec: The input vector.

        Returns:
            vector of results from multiplying the matrix and vector.
        """

        v_res = [0] * self.len_x

        for x in range(self.len_x):
            for y in range(self.len_y):
                v_res[x] += vec[y] * self.M[y][x]

        return v_res

    def __l_make_bipolar(self, vec):
        """
        Utility method to transform a vector to bipolar representation -1,1.
        Args:
            vec: The input vector that will be transformed to bipolar.

        Returns:
            Bipolar representation of the input.
        """

        ret_vec = []

        for item in vec:
            if item == 0:
                ret_vec.append(-1)
            else:
                ret_vec.append(1)

        return ret_vec

    def __threshold(self, vec):
        """
        Utility method to transform a vector to 0,1 representation.
        Args:
            vec: Input vector

        Returns:
            Transformed vector to 0,1 encoding.
        """

        ret_vec = []

        for i in vec:
            if i < 0:
                ret_vec.append(0)
            else:
                ret_vec.append(1)

        return ret_vec
