#!/usr/bin/env python
# coding: utf-8

from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.multi_fidelity import multi_fidelity_wing
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
from lmgp_pytorch.utils import set_seed
from lmgp_pytorch.optim import noise_tune2

###############Parameters########################
random_state = 4
set_seed(random_state)
qual_index = {10:4}
num={'0': 500, '1': 1000, '2': 1000, '3': 1000}
noise_std={'0': 0.5, '1': 1.0, '2': 1.5, '3': 2.0}
############################ Generate Data #########################################
X, y = multi_fidelity_wing(n = num, noise_std= noise_std, random_state = random_state)
############################## train test split ####################################
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.99, 
    qual_index_val= qual_index, stratify= X[...,list(qual_index.keys())])
############################### Model ##############################################
model = LMGP(Xtrain, ytrain, qual_ind_lev=qual_index)
############################### Fit Model ##########################################
_ = noise_tune2(model, num_restarts= 8)
############################### Score ##############################################
model.score(Xtest, ytest, plot_MSE=True, seperate_levels=True)
############################### Latent Map ##############################################
_ = model.visualize_latent()
model.show()