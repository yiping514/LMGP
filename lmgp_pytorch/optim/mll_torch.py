# Copyright © 2021 by Northwestern University.
# 
# LVGP-PyTorch is copyrighted by Northwestern University. It may be freely used 
# for educational and research purposes by  non-profit institutions and US government 
# agencies only. All other organizations may use LVGP-PyTorch for evaluation purposes 
# only, and any further uses will require prior written approval. This software may 
# not be sold or redistributed without prior written approval. Copies of the software 
# may be made by a user provided that copies are not sold or distributed, and provided 
# that copies are used under the same terms and conditions as agreed to in this 
# paragraph.
# 
# As research software, this code is provided on an "as is'' basis without warranty of 
# any kind, either expressed or implied. The downloading, or executing any part of this 
# software constitutes an implicit agreement to these terms. These terms and conditions 
# are subject to change at any time without prior notice.

import torch
import math
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Dict,List,Optional
from copy import deepcopy
from tqdm import tqdm

def fit_model_torch(
    model,
    model_param_groups:Optional[List]=None,
    lr_default:float=0.1,
    num_iter:int=100,
    num_restarts:int=0,
    break_steps:int = 50) -> float:
    '''Optimize the likelihood/posterior of a standard GP model using `torch.optim.Adam`.

    This is a convenience function that covers many situations for optimizing a standard GP model.
    Note that using L-BFGS through `fit_model_scipy` function is a better optimization strategy.

    :param model: A model instance derived from the `models.GPR` class. Can also pass a instance
        inherting from `gpytorch.models.ExactGP` provided that `num_restarts=0` or 
        the class implements a `.reset_parameters` method.
    :type model: models.GPR

    :param model_param_groups: list of parameters to optimizes or dicts defining parameter
        groups. If `None` is specified, then all parameters with `.requires_grad`=`True` are 
        included. Defaults to `None`.
    :type model_param_groups: list, optional

    :param lr_default: The default learning rate for all parameter groups. To use different 
        learning rates for some groups, specify them `model_param_groups`. 
    :type lr_default: float, optional

    :param num_iter: The number of optimization steps from each starting point. This is the only
        termination criterion for the optimizer.
    :type num_iter: float, optional

    :param num_restarts: The number of times to restart the local optimization from a 
        new starting point. Defaults to 5
    :type num_restarts: int, optional

    :returns: the best (negative) log-likelihood/log-posterior found
    :rtype: float
    '''  
    model.train()
    
    # objective
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    f_inc = math.inf
    current_state_dict = model.state_dict()

    loss_hist_total = []
    for i in range(num_restarts+1):
        optimizer = torch.optim.Adam(
            model.parameters() if model_param_groups is None else model_param_groups, 
            lr=lr_default)
        loss_hist = []
        epochs_iter = tqdm(range(num_iter),desc='Epoch',position=0,leave=True)
        for j in epochs_iter:
            # zero gradients from previous iteration
            optimizer.zero_grad()
            # output from model
            output = model(*model.train_inputs)
            # calculate loss and backprop gradients
            loss = -mll(output,model.train_targets)
            loss.backward()
            optimizer.step()

            acc_loss = loss.item()
            desc = f'Epoch {j} - loss {acc_loss:.4f}'
            epochs_iter.set_description(desc)
            epochs_iter.update(1)
            loss_hist.append(acc_loss)

            if j > break_steps and j%break_steps == 0:
                if ( (torch.mean(torch.Tensor(loss_hist)[j-break_steps:j]) - loss_hist[j]) <= 0 ):
                    break
        
        loss_hist_total.append(loss_hist)

        if loss.item()<f_inc:
            current_state_dict = deepcopy(model.state_dict())
            f_inc = loss.item()
        
        if i < num_restarts:
            model.reset_parameters()
    
    model.load_state_dict(current_state_dict)

    return f_inc, loss_hist_total