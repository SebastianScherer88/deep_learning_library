# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:11:32 2018

@author: bettmensch
"""

# This script tests the GeneWeightTranslator, GA and FFNetwork classes in one
# pipeline


#------------------------
# [0] Import dependencies
#------------------------

from diy_deep_learning_library import GA, GeneWeightTranslator, FFNetwork
import numpy as np

#------------------------
# [1] Build network
#------------------------

neural_net = FFNetwork()

# bogus data sample; needed to fixate net
X_sample = np.random.normal(size=(10,400))

n1 = 300
dropoutRate1 = 0.4
n2 = 300
dropoutRate2 = 0.4
n3 = 10

neural_net.addFCLayer(n1,activation='tanh')
neural_net.addDropoutLayer(dropoutRate1)
neural_net.addFCLayer(n2,activation='tanh')
neural_net.addDropoutLayer(dropoutRate2)
neural_net.addFCLayer(n3,activation='softmax')

neural_net.fixateNetwork(X_sample)

print(neural_net)

#------------------------
# [2] Build network weights <-> genes translator
#------------------------

translator = GeneWeightTranslator(neural_net)

# get gene initializer
initializer = translator.initialize_genes

# --- test some translator functionality
# attributes
translator.dna_seq_len == 400 * 300 + 300 + 300 ** 2 + 300 + 300 * 10 + 10

# initializer: initializes network weights and returns (last) corresponding gene
init_genes = initializer(10)
first_gene = init_genes[0,:]
last_gene = init_genes[-1,:]

# check current weights == last gene -> True
init_weights = translator.get_current_weights()
init_weights_converted = translator.weights_to_gene(init_weights)
(init_weights_converted == first_gene).all() # False
(init_weights_converted == last_gene).all() # True

# set weights functionality: use gene set weights, get current weights, convert back to gene, check against original -> True
first_weight = translator.gene_to_weights(first_gene)
translator.set_current_weights(first_weight)
current_weight = translator.get_current_weights()
current_gene = translator.weights_to_gene(current_weight)
(current_gene == first_gene).all() # True