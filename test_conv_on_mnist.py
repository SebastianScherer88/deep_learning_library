# -*- coding: utf-8 -*-
"""
Created on Sun May  6 11:55:22 2018

@author: bettmensch
"""

# This script tests 
# - the fully connected
# - the dropout
# - the convolutional
# - the conv. to fc. reshape
# layer classes from the deep learning library classes by building a
# classification model, training it on the mnist data set and evaluating
# its performance

# [1] imports from library and python lbs
# [2] Build convolitional network
# [3] Train convolutional network on mnist
# [4] Evaluate convolutional network on mnist test data

  
# ---------------------------------
# [1] Imports
# ---------------------------------

import numpy as np
import mnist_classes
import diy_deep_learning_library
import importlib
# for debugging: importlib.reload(diy_deep_learning_library)

# ---------------------------------
# [2] Build convolitional network
# ---------------------------------

# get data
X_train, X_test, y_train, y_test = mnist_classes.get_mnist_data()

print('Shape of X_train:',X_train.shape)
print('Shape of y_train:',y_train.shape)

X_train, X_test = np.array(X_train).reshape(-1,1,28,28),np.array(X_test).reshape(-1,1,28,28)
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

print('Shape of X_train:',X_train.shape)
print('Shape of y_train:',y_train.shape)

# build network
neuralNet = diy_deep_learning_library.FFNetwork(2)

kernelSize1 = 5
channels1 = 6
stride1 = 1
padding1 = 'valid'

poolingSize2 = 2
stride2 = 2
padding2 = 'valid'

kernelSize3 = 5
channels3 = 16
stride3 = 1
padding3 = 'valid'

poolingSize4 = 2
stride4 = 2
padding4 = 'valid'

#n5 = 400

n6 = 84

n7 = 10

neuralNet.addConvLayer(kernelHeight=kernelSize1,
                       kernelWidth=kernelSize1,
                       channels=channels1,
                       stride=stride1,
                       padding=padding1,
                       activation='tanh')
neuralNet.addPoolingLayer(poolingHeight=poolingSize2,
                       poolingWidth=poolingSize2,
                       stride=stride2,
                       padding=padding2,
                       poolingType='max')
neuralNet.addConvLayer(kernelHeight=kernelSize3,
                       kernelWidth=kernelSize3,
                       channels=channels3,
                       stride=stride3,
                       padding=padding3,
                       activation='tanh')
neuralNet.addPoolingLayer(poolingHeight=poolingSize4,
                       poolingWidth=poolingSize4,
                       stride=stride4,
                       padding=padding4,
                       poolingType='max')

neuralNet.addFlattenConvLayer()

neuralNet.addFCLayer(n6,activation='tanh')

neuralNet.addFCLayer(n7,activation='softmax')

neuralNet.fixateNetwork(X_train[:10,:,:,:])

print('\n ')
print(neuralNet)

# ---------------------------------
# [3] Train convolitional network on mnist data
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
# [4] Evaluate convolutional network on mnist test data
# ---------------------------------

# evaluate trained model
PTrain = neuralNet.predict(X_train[:5000])
PTest = neuralNet.predict(X_test[:2000])

accuracyTrain = np.sum(PTrain == y_train[:5000]) / len(PTrain)
accuracyTest = np.sum(PTest == y_test[:2000]) / len(PTest)

print('Accuracy on training set:',accuracyTrain)
print('Accuracy on test set:',accuracyTest)