import os
import subprocess 
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
from typing import Dict

parser = argparse.ArgumentParser('M2AX LVGP accuracy')
parser.add_argument('--save_dir',type=str,required=True)
parser.add_argument('--response',type=str,required=True)
parser.add_argument('--train_split',type=float,required=True)
parser.add_argument('--n_jobs',type=int,required=True)
parser.add_argument('--n_repeats',type=int,default=25)
args = parser.parse_args()


save_dir = os.path.join(
    args.save_dir,
    args.response,'train_split_%.1f'%args.train_split
)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#%%
dat = pd.read_csv('M2AX_data.csv')
config = InputSpace()
col_names = ['%s-site element'%l for l in ['M','A','X']]
elems = [
    CategoricalVariable(name=name,levels=dat[name].unique()) \
    for name in col_names
]
config.add_inputs(elems)

all_combs = torch.tensor([
    config.get_array_from_dict(row) for _,row in dat[config.get_variable_names()].iterrows()
]).double()

if args.response == 'Young':
    target = "E (Young's modulus)"
elif args.response == 'Shear':
    target = "G (Shear modulus)"
elif args.response == 'Bulk':
    target = "B (Bulk modulus)"

all_responses = -torch.tensor(dat[target]).double()
#%%
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def main_script(seed):
    num_samples = int(dat.shape[0]*args.train_split)

    save_dir_seed = os.path.join(save_dir,'seed_%d'%seed)
    if not os.path.exists(save_dir_seed):
        os.makedirs(save_dir_seed)
    
    # sample without replacement
    rng = np.random.RandomState(seed)
    idxs_train = rng.choice(all_combs.shape[0],num_samples,replace=False)
    train_x = all_combs[idxs_train,:]
    train_y = all_responses[idxs_train]
    
    idxs_test = np.array([idx for idx in np.arange(all_combs.shape[0]) if idx not in idxs_train])
    test_x = all_combs[idxs_test,:]
    test_y = all_responses[idxs_test]
    # save training data
    np.savetxt(os.path.join(save_dir_seed,'train_idxs.csv'),np.array(idxs_train))

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
    _,nll_inc1 = fit_model_scipy(model1)
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
    nll_inc2,_ = noise_tune(model2)
    fit_time2 = time.time()-start_time

    with torch.no_grad():
        test_mean = model2.predict(test_x,return_std=False)
        
    # print RRMSE
    rrmse2 = (((test_y-test_mean)**2).sum().sqrt()/((test_y-test_y.mean())**2).sum().sqrt()).item()

    # ####
    # # run R script
    # ####
    _ = subprocess.call(['Rscript','R_lvgp.R',save_dir_seed,args.response])
    tmp = pd.read_csv(os.path.join(save_dir_seed,'R_stats.csv'))

    out = {
        'rrmse1':rrmse1,'fit_time1':fit_time1,'nlp_inc1':nll_inc1,
        'rrmse2':rrmse2,'fit_time2':fit_time2,'nlp_inc2':nll_inc2,
        'rrmse_R':tmp['rrmse'].values[0],'fit_time_R':tmp['fit_time'].values[0]
    }
    dump(out,os.path.join(save_dir_seed,'comps.pkl'))

#%%
seeds = np.linspace(100,1000,args.n_repeats).astype(int)

Parallel(n_jobs=args.n_jobs,verbose=0)(
    delayed(main_script)(seed) for seed in seeds
)