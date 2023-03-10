#!/usr/bin/env python
from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.physical import borehole_mixed_variables
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
from lmgp_pytorch.utils import set_seed
from lmgp_pytorch.optim import fit_model_scipy

############################### Paramter of the model #########################
##__###
random_state = 4
set_seed(random_state)
qual_index = {0:5,6:4}
############################ Generate Data #########################################
X, y = borehole_mixed_variables(n = 10000, qual_ind_val= qual_index, random_state = random_state)
############################## train test split ####################################
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.99, 
    qual_index_val= qual_index)
############################### Model ##############################################
# model = LMGP(Xtrain, ytrain, qual_ind_lev=qual_index)
lv_columns=[0]
model = LMGP(Xtrain, ytrain, qual_ind_lev=qual_index, lv_columns=lv_columns)
############################### Fit Model ##########################################
_ = fit_model_scipy(model, bounds=True)
############################### Score ##############################################
model.score(Xtest, ytest, plot_MSE=True)
############################### latent space ########################################
_ = model.visualize_latent()
model.show()