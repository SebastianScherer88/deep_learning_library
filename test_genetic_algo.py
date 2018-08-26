# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 18:51:13 2018

@author: bettmensch
"""

# This is a test script for the genetic algorithm class GA


# --------------------
# [0] Imports
# --------------------

from diy_deep_learning_library import GA
import numpy as np

# --------------------
# [1] Set up maximisation function
# --------------------

# --- util function to generate function to maximise
def get_peak(hight,
             center,
             sigma):
    '''Util function that returns an normal dist - like peak function around
    the specified center (list with x,y arguments), with the specified hight (scalar)
    and the specified standard deviation sigma (scalar).'''
    
    peak_function = lambda x_y: hight * np.exp(-(np.sum(x_y - np.array(center)) ** 2) / (2 * sigma ** 2))
    
    return peak_function

# --- generate function to maximise
#   first peak - center = (2,2), hight = 2, stdev = 0.5
first_peak = get_peak(2,[2,2],0.5)    

#   second peak - center = (5,8), hight = 4, stdev = 0.5
second_peak = get_peak(4,[5,8],0.5)    
    
#   third peak - center = (7,4), hight = 8, stdev = 0.5
third_peak = get_peak(8,[7,4],0.5)

#   function to maximise
def f_max(x_y):    
    return first_peak(x_y) + second_peak(x_y) + third_peak(x_y)

x1 = np.array([2,2])
x2 = np.array([5,8])
x3 = np.array([7,4])

f_max(x1)
f_max(x2)
f_max(x3)

# --------------------
# [2] 
# --------------------
