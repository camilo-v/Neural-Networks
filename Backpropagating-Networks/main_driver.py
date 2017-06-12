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
import numpy as np
np.set_printoptions(linewidth=500)
import time

#   Custom Modules
sys.path.append(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Multi-Layer-Perceptron'))

import utility_functions as uf
import MultiLayerPerceptron as mlp


# ------------------------------------------------------ Main ---------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] " + "Project 2, Backpropagation Networks" + "" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

#   Base directory for project modules, files, etc.
baseDirectoryForProject = "/Users/camilo/Documents/Development/GitHub/Neural-Networks/Backpropagating-Networks"
#   Training Data for letters A and E.

filePathForTraining = baseDirectoryForProject + "/data/letters/data-training-all.txt"

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
trainingTargetValues_y_Set_1, testingSet1FeatureMatrix_X = uf.readTestingData( pathForTestingSet_1 )
trainingTargetValues_y_Set_2, testingSet2FeatureMatrix_X = uf.readTestingData( pathForTestingSet_2 )
trainingTargetValues_y_Set_3, testingSet3FeatureMatrix_X = uf.readTestingData( pathForTestingSet_3 )

#   Data structures for testing sets (feature matrices only)
df_testingSet_1_y = pd.DataFrame(trainingTargetValues_y_Set_1).as_matrix()
df_testingSet_2_y = pd.DataFrame(trainingTargetValues_y_Set_2).as_matrix()
df_testingSet_3_y = pd.DataFrame(trainingTargetValues_y_Set_3).as_matrix()

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


#
#   Initialize the Multi-Layered Perceptron object.
#

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation, Training MLP..." + "" )

momentumConstant = 1    # Not used in this step, but required for the constructor.
numberOfEpochs = 1000
numberOfHiddenUnits = 25

nn = mlp.MultiLayerPerceptron(  n_output=3,
                                n_features=df_X.shape[1],
                                n_hidden=numberOfHiddenUnits,
                                l2=0.1,
                                l1=0.0,
                                epochs=numberOfEpochs,
                                eta=0.001,
                                alpha=momentumConstant,
                                decrease_const=0.00001,
                                minibatches=1,
                                shuffle=True,
                                random_state=1,
                                useMomentum=False,
                                useNguyenWidrow=False,
                                destroyWeights=True,
                                destroyAmount=40 )

#   Fit the training data using the initialized MLP object.
nn.fit( df_X, df_y, print_progress=True)


#   Diagnostic plots
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 100])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.title('Backpropagation')
plt.show()

#  Weight Diagnostic plots
# plt.plot(range(len(nn.cost_)), nn.cost_)
# plt.ylim([0, 300])
# plt.ylabel('Evolution of Weights & Bias')
# plt.xlabel('Patterns')
# plt.tight_layout()
# plt.title('Backpropagation with Nguyen-Widrow & Momentum')
# plt.show()


#   Meassure the accuracy of the training step
backpropTraining_pred = nn.predict( df_X )
accuracyForBackPropTraining = (np.sum(df_y.T == backpropTraining_pred) / df_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation" +
       " Training accuracy: %.2f%%" % (accuracyForBackPropTraining * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

#
#   Test the trained MLP with the testing datasets.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation, Testing Set 1..." + "" )

backpropTest1_pred = nn.predict( df_testingSet_1_X )
accuracyForBackPropTest1 = (np.sum(df_testingSet_1_y.T == backpropTest1_pred) / df_testingSet_1_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation" +
       " Testing Set 1 Accuracy: %.2f%%" % (accuracyForBackPropTest1 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )


print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation, Testing Set 2..." + "" )

backpropTest2_pred = nn.predict( df_testingSet_2_X )
accuracyForBackPropTest2 = (np.sum(df_testingSet_2_y.T == backpropTest2_pred) / df_testingSet_2_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation" +
        " Testing Set 2 Accuracy: %.2f%%" % (accuracyForBackPropTest2 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation, Testing Set 3..." + "" )

backpropTest3_pred = nn.predict( df_testingSet_3_X )
accuracyForBackPropTest3 = (np.sum(df_testingSet_3_y.T == backpropTest3_pred) / df_testingSet_3_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation" +
        " Testing Set 3 Accuracy: %.2f%%" % (accuracyForBackPropTest3 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )


# --------------------------------------------- Backpropagation with Momemtum -----------------------------------------
#
#      The following section trains a Multi Layered Perceptron (MLP) with the Backpropagation  plus momemtum algorithm.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation with Momemtum..." + "" )


#
#   Initialize the Multi-Layered Perceptron object for use with Momentum by setting the flag "useMomentum" to True.
#

momentumConstant2 = 0.9
numberOfEpochs2 = 1000
numberOfHiddenUnits2 = 25

nn_2 = mlp.MultiLayerPerceptron(    n_output=3,
                                    n_features=df_X.shape[1],
                                    n_hidden=numberOfHiddenUnits2,
                                    l2=0.1,
                                    l1=0.0,
                                    epochs=numberOfEpochs2,
                                    eta=0.001,
                                    alpha=momentumConstant2,
                                    decrease_const=0.00001,
                                    minibatches=1,
                                    shuffle=True,
                                    random_state=1,
                                    useMomentum=True,
                                    useNguyenWidrow=False,
                                    destroyWeights=True,
                                    destroyAmount=40 )

#   Fit the training data using the initialized MLP object.
nn_2.fit( df_X, df_y, print_progress=True)

#   Diagnostic plots
plt.plot(range(len(nn_2.cost_)), nn_2.cost_)
plt.ylim([0, 100])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.title('Backpropagation with Momentum = ' + str(momentumConstant2))

plt.show()

#
#   Meassure the accuracy of the training step
backpropWithMomentumTraining_pred = nn_2.predict( df_X )
accuracyForBackPropWithMomentumTraining = (np.sum(df_y.T == backpropWithMomentumTraining_pred) / df_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Momentum" +
       " Training Set Accuracy: %.2f%%" % (accuracyForBackPropWithMomentumTraining * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

#
#   Test the trained MLP with the testing datasets for Backpropagation with Momentum.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Momentum, Testing Set 1..." + "" )

backpropWithMomentumTest1_pred = nn_2.predict( df_testingSet_1_X )
accuracyForBackPropWithMomentumTest1 = (np.sum(df_testingSet_1_y.T == backpropWithMomentumTest1_pred) /
                                        df_testingSet_1_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Momentum" +
       " Testing Set 1 Accuracy: %.2f%%" % (accuracyForBackPropWithMomentumTest1 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )


print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Momentum, Testing Set 2..." + "" )

backpropWithMomentumTest2_pred = nn_2.predict( df_testingSet_2_X )
accuracyForBackPropWithMomentumTest2 = (np.sum(df_testingSet_2_y.T == backpropWithMomentumTest2_pred) /
                                        df_testingSet_2_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Momentum" +
       " Testing Set 2 Accuracy: %.2f%%" % (accuracyForBackPropWithMomentumTest2 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Momentum, Testing Set 3..." + "" )

backpropWithMomentumTest3_pred = nn_2.predict( df_testingSet_3_X )
accuracyForBackPropWithMomentumTest3 = (np.sum(df_testingSet_3_y.T == backpropWithMomentumTest3_pred) /
                                        df_testingSet_3_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Momentum" +
       " Testing Set 2 Accuracy: %.2f%%" % (accuracyForBackPropWithMomentumTest3 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )



# ------------------------------------------- Backpropagation with Nguyen-Widrow --------------------------------------
#
#   The following section trains a Multi Layered Perceptron (MLP) with Backpropagation and Nguyen-Widrow initialization
#   algorithm, but with no momemtum.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation with Nguyen-Widrow..." + "" )

#
#   Initialize the Multi-Layered Perceptron object for use with Nguyen-Widrow initialization by setting the flag
#   "useNguyenWidrow" to True.  Note that here we do not use Momentum, so we set "useMomentum" to False.
#

momentumConstant3 = 1   #   As before, set but not used.
numberOfEpoch3 = 1000
numberOfHiddenUnits3 = 25

nn_3 = mlp.MultiLayerPerceptron(    n_output=3,
                                    n_features=df_X.shape[1],
                                    n_hidden=numberOfHiddenUnits3,
                                    l2=0.1,
                                    l1=0.0,
                                    epochs=numberOfEpoch3,
                                    eta=0.001,
                                    alpha=momentumConstant3,
                                    decrease_const=0.00001,
                                    minibatches=1,
                                    shuffle=True,
                                    random_state=1,
                                    useMomentum=False,
                                    useNguyenWidrow=True,
                                    destroyWeights=True,
                                    destroyAmount=40 )

#   Fit the training data using the initialized MLP object.
nn_3.fit( df_X, df_y, print_progress=True)

#   Diagnostic plots
plt.plot(range(len(nn_3.cost_)), nn_3.cost_)
plt.ylim([0, 100])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.title('Backpropagation with Nguyen-Widrow')

plt.show()


#
#   Meassure the accuracy of the training step
backpropWithNguyenWidrowTraining_pred = nn_3.predict( df_X )
accuracyForBackPropWithNguyenWidrowTraining = (np.sum(df_y.T == backpropWithNguyenWidrowTraining_pred) / df_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Nguyen-Widrow" +
       " Training Set Accuracy: %.2f%%" % (accuracyForBackPropWithNguyenWidrowTraining * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

#
#   Test the trained MLP with the testing datasets for Backpropagation with Momentum.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Nguyen-Widrow, Testing Set 1..." + "" )

backpropWithNguyenWidrowTest1_pred = nn_3.predict( df_testingSet_1_X )
accuracyForBackPropWithNguyenWidrowTest1 = (np.sum(df_testingSet_1_y.T == backpropWithNguyenWidrowTest1_pred) /
                                            df_testingSet_1_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Nguyen-Widrow" +
       " Testing Set 1 Accuracy: %.2f%%" % (accuracyForBackPropWithNguyenWidrowTest1 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )


print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Nguyen-Widrow, Testing Set 2..." + "" )

backpropWithNguyenWidrowTest2_pred = nn_3.predict( df_testingSet_2_X )
accuracyForBackPropWithNguyenWidrowTest2 = (np.sum(df_testingSet_2_y.T == backpropWithNguyenWidrowTest2_pred) /
                                            df_testingSet_2_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Nguyen-Widrow" +
       " Testing Set 2 Accuracy: %.2f%%" % (accuracyForBackPropWithNguyenWidrowTest2 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Nguyen-Widrow, Testing Set 3..." + "" )

backpropWithNguyenWidrowTest3_pred = nn_3.predict( df_testingSet_3_X )
accuracyForBackPropWithNguyenWidrowTest3 = (np.sum(df_testingSet_3_y.T == backpropWithNguyenWidrowTest3_pred) /
                                            df_testingSet_3_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Nguyen-Widrow" +
       " Testing Set 2 Accuracy: %.2f%%" % (accuracyForBackPropWithNguyenWidrowTest3 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )



# -------------------------------------- Backpropagation with Momemtum and Nguyen-Widrow ------------------------------
#
#   The following section trains a Multi Layered Perceptron (MLP) with Backpropagation and Momemtum, along with the
#   Nguyen-Widrow initialization algorithm.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] ----------------------------------" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation with Momemtum and Nguyen-Widrow..." + "" )


momentumConstant4 = 0.9
numberOfEpoch4 = 1000
numberOfHiddenUnits4 = 25

nn_4 = mlp.MultiLayerPerceptron(    n_output=3,
                                    n_features=df_X.shape[1],
                                    n_hidden=numberOfHiddenUnits3,
                                    l2=0.1,
                                    l1=0.0,
                                    epochs=numberOfEpoch3,
                                    eta=0.001,
                                    alpha=momentumConstant3,
                                    decrease_const=0.00001,
                                    minibatches=1,
                                    shuffle=True,
                                    random_state=1,
                                    useMomentum=True,
                                    useNguyenWidrow=True,
                                    destroyWeights=True,
                                    destroyAmount=40 )

#   Fit the training data using the initialized MLP object.
nn_4.fit( df_X, df_y, print_progress=True)

#   Diagnostic plots
plt.plot(range(len(nn_4.cost_)), nn_4.cost_)
plt.ylim([0, 100])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.title('Backpropagation with Momentum and Nguyen-Widrow')

plt.show()

#
#   Meassure the accuracy of the training step
backpropWithMomentumAndNguyenWidrowTraining_pred = nn_4.predict( df_X )
accuracyForBackPropWithMomentumAndNguyenWidrowTraining = (np.sum(df_y.T == backpropWithMomentumAndNguyenWidrowTraining_pred) /
                                               df_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Momentum & Nguyen-Widrow" +
       " Training Set Accuracy: %.2f%%" % (accuracyForBackPropWithMomentumAndNguyenWidrowTraining * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

#
#   Test the trained MLP with the testing datasets for Backpropagation with Momentum.
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Nguyen-Widrow, Testing Set 1..." + "" )

backpropWithMomentumAndNguyenWidrowTest1_pred = nn_4.predict( df_testingSet_1_X )
accuracyForBackPropWithMomentumAndNguyenWidrowTest1 = (np.sum(df_testingSet_1_y.T == backpropWithMomentumAndNguyenWidrowTest1_pred) /
                                            df_testingSet_1_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Momentum & Nguyen-Widrow" +
       " Testing Set 1 Accuracy: %.2f%%" % (accuracyForBackPropWithMomentumAndNguyenWidrowTest1 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )


print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Nguyen-Widrow, Testing Set 2..." + "" )

backpropWithMomentumAndNguyenWidrowTest2_pred = nn_4.predict( df_testingSet_2_X )
accuracyForBackPropWithMomentumAndNguyenWidrowTest2 = (np.sum(df_testingSet_2_y.T == backpropWithMomentumAndNguyenWidrowTest2_pred) /
                                            df_testingSet_2_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Momentum & Nguyen-Widrow" +
       " Testing Set 2 Accuracy: %.2f%%" % (accuracyForBackPropWithMomentumAndNguyenWidrowTest2 * 100) )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" +
       " Backpropagation w/ Nguyen-Widrow, Testing Set 3..." + "" )

backpropWithMomentumAndNguyenWidrowTest3_pred = nn_4.predict( df_testingSet_3_X )
accuracyForBackPropWithMomentumAndNguyenWidrowTest3 = (np.sum(df_testingSet_3_y.T == backpropWithMomentumAndNguyenWidrowTest3_pred) /
                                            df_testingSet_3_X.shape[0])

print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" + " Backpropagation w/ Momentum & Nguyen-Widrow" +
       " Testing Set 2 Accuracy: %.2f%%" % (accuracyForBackPropWithMomentumAndNguyenWidrowTest3 * 100) )


print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )




# -------------------------------------------------- End --------------------------------------------------------------
#
#
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ] Done." )
print( "[ " + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + " ]" )
