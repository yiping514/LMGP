#!/usr/bin/env python
from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.physical import borehole_mixed_variables
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
from lmgp_pytorch.preprocessing.one_hot_encoding import one_hot_encoding
from lmgp_pytorch.optim import fit_model_scipy
import numpy as np
import torch
from lmgp_pytorch.utils import set_seed
from lmgp_pytorch.preprocessing import standard 
from sklearn.model_selection import train_test_split
############################### Paramter of the model #########################
##__###
random_state = 4
set_seed(random_state)
qual_index ={0:5,6:4}
############################ Generate Data #########################################
X, y = borehole_mixed_variables(n = 10000, qual_ind_val= qual_index, random_state = random_state)
qual_index_list = list(qual_index.keys())
all_index = set(range(X.shape[-1]))
quant_index = list(all_index.difference(qual_index_list))
################################ One_Hot_Encoding ##############################################

X_encoded=one_hot_encoding(X,qual_index)
############################## train test split ####################################


# Split test and train
Xtrain, Xtest, ytrain, ytest = train_test_split(X_encoded, y, 
    test_size= .99)
# Standard
Xtrain, Xtest, mean_train, std_train = standard(Xtrain = Xtrain, 
    quant_index = quant_index, Xtest = Xtest)


############################### Model ##############################################
model = LMGP(Xtrain, torch.from_numpy(ytrain))
############################### Fit Model ##########################################
_ = fit_model_scipy(model, bounds=False)
############################### Score ##############################################
model.score(Xtest, torch.from_numpy(ytest), plot_MSE=True)
############################### latent space ########################################
_ = model.visualize_latent()
model.show()