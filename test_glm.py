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
predictor_means = np.array([7,3,-0.6,15,0.2]).reshape(1,-1)
predictor_sds = np.array([0.21,0.1,0.3,4,0.8]).reshape(1,-1)

# sample predictors from standard normal and then adjust & stretch to create
# different profiles
X = np.random.randn(n,5) * predictor_sds + predictor_means
                   
# set true linear coefficients
beta_true = np.array([4.2,-2,3,-0.7,5.8]).reshape(1,-1)
intercept_true = jitter_mean = 8

# prep some random noise from uniform [0,1] * skew + mean
noise_stretch = 1.6
noise = np.random.rand(n,1) * noise_stretch
                               
# create response as actual samples from constructed poissons, using the specs
# set so far
Y_means = np.dot(X,beta_true.T) + intercept_true + noise
Y = np.random.poisson(lam=Y_means)

assert(X.shape[0] == Y.shape[0])

# split data for modelling
X_train, Y_train = X[1:int(n*0.8),], Y[1:int(n*0.8),]
X_test, Y_test = X[int(n*0.8)+1,], Y[int(n*0.8)+1,]

# scale predictors
scaler = StandardScaler.fit(X_train)

X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# [2] Train model