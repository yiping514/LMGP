import os
import argparse
import torch
import time
import random
import numpy as np
import pandas as pd
from joblib import dump,Parallel,delayed

from lmgp_pytorch.models import LVGPR
from lmgp_pytorch.optim import fit_model_scipy
from lmgp_pytorch.optim.mll_noise_tune import noise_tune
from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
from lmgp_pytorch.utils.input_space import InputSpace

# functions
import functions
# confgiurations
import configs

parser = argparse.ArgumentParser('LVGP tests on math functions')
parser.add_argument('--which_func',type=str,required=True)
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--num_samples',type=int,required=True)
parser.add_argument('--n_jobs',type=int,required=True)
parser.add_argument('--n_repeats',type=int,default=25)
args = parser.parse_args()

save_dir = os.path.join(
    args.save_dir,
    '%s/%d_samples'%(args.which_func,args.num_samples)
)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#%% configuration object and function definition
config = getattr(configs,args.which_func)()
obj = getattr(functions, args.which_func)

#%% test data
test_x = torch.from_numpy(config.random_sample(np.random.RandomState(456),1000))
test_y = [None]*test_x.shape[0]

for i,x in enumerate(test_x):
    test_y[i] = obj(config.get_dict_from_array(x.numpy()))
    
# create tensor objects
test_y = torch.tensor(test_y).to(test_x)

# save it for reference
np.savetxt(os.path.join(save_dir,'../','test_x.csv'),test_x.numpy())
np.savetxt(os.path.join(save_dir,'../','test_y.csv'),test_y.numpy())
np.savetxt(os.path.join(save_dir,'../','qual_index.csv'),np.array(config.qual_index))

#%%
# training data
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def main_script(seed):
    save_dir_seed = os.path.join(save_dir,'seed_%d'%seed)
    if not os.path.exists(save_dir_seed):
        os.makedirs(save_dir_seed)
    
    train_x = torch.from_numpy(config.latinhypercube_sample(np.random.RandomState(seed),args.num_samples))
    train_y = [None]*args.num_samples

    for i,x in enumerate(train_x):
        train_y[i] = obj(config.get_dict_from_array(x.numpy()))
        
    # create tensor objects
    train_y = torch.tensor(train_y).to(train_x)

    np.savetxt(os.path.join(save_dir_seed,'train_x.csv'),train_x.numpy())
    np.savetxt(os.path.join(save_dir_seed,'train_y.csv'),train_y.numpy())

    # ####
    # # optimizing noise variance along with the other hyperparameters
    # ####
    
    set_seed(seed)
    model1 = LVGPR(
        train_x=train_x,
        train_y=train_y,
        quant_correlation_class='RBFKernel',
        qual_index=config.qual_index,
        quant_index=config.quant_index,
        num_levels_per_var=list(config.num_levels.values()),
    ).double()

    start_time = time.time()
    _,nll_inc1 = fit_model_scipy(model1,num_restarts=24)
    fit_time1 = time.time()-start_time

    with torch.no_grad():
        test_mean = model1.predict(test_x,return_std=False)
        
    # print RRMSE
    rrmse1 = (((test_y-test_mean)**2).sum().sqrt()/((test_y-test_y.mean())**2).sum().sqrt()).item()

    # ####
    # # using the nugget tuning algorithm
    # ####
    set_seed(seed)
    model2 = LVGPR(
        train_x=train_x,
        train_y=train_y,
        quant_correlation_class='RBFKernel',
        qual_index=config.qual_index,
        quant_index=config.quant_index,
        num_levels_per_var=list(config.num_levels.values()),
    ).double()

    start_time = time.time()
    nll_inc2,_ = noise_tune(model2,num_restarts=15)
    fit_time2 = time.time()-start_time

    with torch.no_grad():
        test_mean = model2.predict(test_x,return_std=False)
        
    # print RRMSE
    rrmse2 = (((test_y-test_mean)**2).sum().sqrt()/((test_y-test_y.mean())**2).sum().sqrt()).item()

    out = {
        'rrmse1':rrmse1,'fit_time1':fit_time1,'nlp_inc1':nll_inc1,
        'rrmse2':rrmse2,'fit_time2':fit_time2,'nlp_inc2':nll_inc2,
    }
    dump(out,os.path.join(save_dir_seed,'comps.pkl'))

#%%
seeds = np.linspace(100,1000,args.n_repeats).astype(int)

Parallel(n_jobs=args.n_jobs,verbose=0)(
    delayed(main_script)(seed) for seed in seeds
)