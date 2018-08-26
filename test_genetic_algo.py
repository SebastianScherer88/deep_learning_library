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
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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
    
    peak_function = lambda x_y: hight * np.exp((-np.sum((x_y - np.array(center)) ** 2)) / (2 * sigma ** 2))
    
    return peak_function

# --- generate function to maximise
#   first peak - center = (2,2), hight = 2, stdev = 0.5
first_peak = get_peak(0.6,[2,2],1)    

#   second peak - center = (5,8), hight = 4, stdev = 0.5
second_peak = get_peak(0.8,[4,8],1)    
    
#   third peak - center = (7,4), hight = 8, stdev = 0.5
third_peak = get_peak(0.4,[7,4],1)

#   function to maximise
def f_max(x_y):    
    return first_peak(x_y) + second_peak(x_y) + third_peak(x_y)

# --------------------
# [2]  Visualize maximization function
# --------------------

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 10, 0.25)
Y = np.arange(0, 10, 0.25)
X, Y = np.meshgrid(X, Y)

Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        xy_ij = np.array([X[i,j],Y[i,j]])
        Z[i,j] = f_max(xy_ij)
        
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# --------------------
# [3] Maximise via genetic algorithm
# --------------------

# --- specs
#    needs to be of the form initializer(n_pop) and return an array of shape
#    (n_pop, dna_seq_len)
initializer = lambda n: np.random.normal(scale=20,size=(n,2))
# mutation
mutation_rate = 0.2
# max gens
max_gens = 100
# population size
n_pop = 60

# --- evolution
#   get algo object
genetic_algo = GA(2,
                  initializer = initializer)
#   evolve
recent_gen = genetic_algo.evolve(f_max,
                                 n_pop = n_pop,
                                 mutation_rate = mutation_rate,
                                 max_gens = max_gens)

# --------------------
# [4] Analyse results
# --------------------

# --- get stats
historic_averages = genetic_algo.population_history.groupby('n_gen').mean()[['score']]
historic_maxes = genetic_algo.population_history.groupby('n_gen').max()[['score']]

history_info = pd.DataFrame(columns=['n_gen','average_score','max_score'])
history_info['n_gen'] = [i+1 for i in range(max_gens)]
history_info['average_score'] = historic_averages['score']
history_info['max_score'] = historic_maxes['score']

# --- visualize stats
#   history
plt.plot(history_info['max_score']) # blue curve
plt.plot(history_info['average_score']) # orange curve
plt.show()
#   winner gene
elite_gene = genetic_algo.population_history.sort_values('score')[['gene1','gene2','score']].iloc[-1]
print(elite_gene)