# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:29:39 2019

@author: bettmensch
"""

# This script runs a test on the ne GLM class on some artificial poisson
# data.

# [0] Imports and dependencies
import os
os.chdir('C:\\Users\\bettmensch\\GitReps\\deep_learning_library')
from diy_deep_learning_library import GLM
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler

# [1] Create data set
np.random.seed(seed = 10)
n = 10000

# generate some predictors
predictor_means = np.array([-7,3,-0.6,1.5,0.2]).reshape(1,-1)
predictor_sds = np.array([2,0.1,0.3,4,0.8]).reshape(1,-1)

# sample predictors from standard normal and then adjust & stretch to create
# different profiles
X = np.random.randn(n,5) * predictor_sds + predictor_means
                   
# set true linear coefficients
beta_true = np.array([1.2,2,-0.3,-7,-10]).reshape(1,-1)
intercept_true = jitter_mean = -2

# prep some random noise from uniform [0,1] * skew + mean
noise_stretch = 1.6
noise = (np.random.rand(n,1) - 0.5) * noise_stretch
                               
# create response as actual samples from constructed poissons, using the specs
# set so far
Y_means = np.exp(np.dot(X,beta_true.T) + intercept_true) #+ noise

# large values create issues when sampling from the poisson, so filter those out
mean_max = 10
useful_obs_index = (Y_means < mean_max).reshape(-1)
Y_means = Y_means[useful_obs_index,:]
Y = np.random.poisson(lam=Y_means)

X = X[useful_obs_index,:]

n_new = Y_means.shape[0]
#Y = Y_means

assert(X.shape[0] == Y.shape[0])

# split data for modelling
X_train, Y_train = X[1:int(n_new*0.8),], Y[1:int(n_new*0.8),]
X_test, Y_test = X[int(n_new*0.8)+1:,], Y[int(n_new*0.8)+1:,]

# scale predictors
scaler = StandardScaler().fit(X_train)

X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# [2] Train model

nEpochs=30
batchSize=20
optimizer='sgd'
eta=0.01
gamma=0.99
epsilon=0.0000001
lamda=0
displaySteps=25
  
glm_poisson = GLM("poisson")

glm_poisson.trainGLM(X=X_train_st,
                     Y=Y_train,
                     nEpochs=nEpochs,
                     batchSize=batchSize,
                     optimizer=optimizer,
                     eta=eta,
                     gamma=gamma,
                     epsilon=epsilon,
                     lamda=lamda,
                     displaySteps=displaySteps)