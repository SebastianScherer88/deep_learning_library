# -*- coding: utf-8 -*-
"""
Created on Sun May  6 11:45:34 2018

@author: bettmensch
"""

# This script tests the fully connected and dropout layer classes from the 
# deep learning library classes by building a
# classification model, training it on the mnist data set and evaluating
# its performance

# [1] imports from library and python lbs
# [2] Build fullly connected net
# [3] Train fully connected net on mnist data set
# [4] Evaluate fully connected net on mnist data test set

# ---------------------------------
# [1] Imports
# ---------------------------------

import os
os.chdir('C:\\Users\\bettmensch\\GitReps\\deep_learning_library')
import numpy as np
import mnist_classes
import diy_deep_learning_library
import importlib
# for debugging: importlib.reload(diy_deep_learning_library)

# ---------------------------------
# [2] Build fullly connected net
# ---------------------------------

# get data
X_train, X_test, y_train, y_test = mnist_classes.get_mnist_data()

X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

# build network
neuralNet = diy_deep_learning_library.FFNetwork()

n1 = 300
dropoutRate1 = 0.4
n2 = 300
dropoutRate2 = 0.4
n3 = 10

neuralNet.addFCLayer(n1,activation='tanh')
neuralNet.addDropoutLayer(dropoutRate1)
neuralNet.addFCLayer(n2,activation='tanh')
neuralNet.addDropoutLayer(dropoutRate2)
neuralNet.addFCLayer(n3,activation='softmax')

neuralNet.fixateNetwork(X_train[:10,:])

print(neuralNet)

# ---------------------------------
# [3] Train fully connected net on mnist data set
# ---------------------------------

# train network
nEpochs = 2
learning_rate = 0.5
regularization_param = 0.1
momentum_param = 0.3
optimizer = 'sgd'
batchSize = 50
displaySteps = 50
oneHotY = True

neuralNet.trainNetwork(X_train,y_train,
                       nEpochs=nEpochs,batchSize=batchSize,
                       optimizer=optimizer,eta=learning_rate,lamda=regularization_param,gamma=momentum_param,
                       displaySteps=displaySteps,oneHotY=oneHotY)

# ---------------------------------
# [4] Evaluate fully connected net on mnist data test set
# ---------------------------------

# evaluate trained model
PTrain = neuralNet.predict(X_train)
PTest = neuralNet.predict(X_test)

accuracyTrain = np.sum(PTrain == y_train) / len(PTrain)
accuracyTest = np.sum(PTest == y_test) / len(PTest)

print('Accuracy on training set:',accuracyTrain)
print('Accuracy on test set:',accuracyTest)