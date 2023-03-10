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

from cmath import inf
from pickle import NONE
import torch
import numpy as np
from gpytorch import settings as gptsettings
from gpytorch.utils.errors import NanError,NotPSDError
from scipy.optimize import minimize,OptimizeResult
from collections import OrderedDict
from functools import reduce
from joblib import Parallel,delayed
from joblib.externals.loky import set_loky_pickler
from typing import Dict,List,Tuple,Optional,Union
from copy import deepcopy

#####################Optimization###########################
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
#######################################################

from lmgp_pytorch.utils.interval_score import interval_score

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}

def marginal_log_likelihood(model,add_prior:bool,regularization_parameter=[0,0]):
    output = model(*model.train_inputs)
    out = model.likelihood(output).log_prob(model.train_targets)
    if add_prior:
        # add priors
        for _, module, prior, closure, _ in model.named_priors():
            out.add_(prior.log_prob(closure(module)).sum())
    temp = 0
    temp_1=0
    for name, param in model.named_parameters():
        string_list = ['fci', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h8','h9', 'h10', 'h11', 'h12','fce']
        if name in string_list:
            temp += torch.norm(param)
            temp_1 += torch.sum(torch.abs(param))
        elif name in ['nn_model.' + str + '.bias' for str in string_list]:
            temp += torch.norm(param)
            temp_1 += torch.sum(torch.abs(param))

    out -= regularization_parameter[0]*temp_1 + regularization_parameter[1]* temp
    score, accuracy = interval_score(output.mean + 1.96 * output.variance.sqrt(), output.mean - 1.96 * output.variance.sqrt(), model.y_scaled)
    return out - 0.0 * score #- torch.exp(model.interval_alpha) * score

class MLLObjective:
    """Helper class that wraps MLE/MAP objective function to be called by scipy.optimize.

    :param model: A :class:`models.GPR` instance whose likelihood/posterior is to be 
        optimized.
    :type model: models.GPR
    """
    def __init__(self,model,add_prior,regularization_parameter):
        self.model = model 
        self.add_prior = add_prior
        self.regularization_parameter=regularization_parameter

        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        self.param_shapes = OrderedDict()
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                if len(parameters[n].size()) > 0:
                    self.param_shapes[n] = parameters[n].size()
                else:
                    self.param_shapes[n] = torch.Size([1])
    
    def pack_parameters(self) -> np.ndarray:
        """Returns the current hyperparameters in vector form for the scipy optimizer

        :return Current hyperparameters in a 1-D array representation
        :rtype: np.ndarray
        """
        parameters = OrderedDict([
            (n,p) for n,p in self.model.named_parameters() if p.requires_grad
        ])
        
        return np.concatenate([parameters[n].cpu().data.numpy().ravel() for n in parameters])
    
    def unpack_parameters(self, x:np.ndarray) -> torch.Tensor:
        """Convert hyperparameters specifed as a 1D array to a named parameter dictionary
        that can be imported by the model

        :param x: Hyperparameters in flattened vector form
        :type x: np.ndarray

        :returns: A dictionary of hyperparameters
        :rtype: Dict
        """
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param).to(**tkwargs)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self) -> None:
        """Concatenate gradients from the parameters to 1D numpy array
        """
        grads = []
        for name,p in self.model.named_parameters():
            if p.requires_grad:
                grad = p.grad.cpu().data.numpy()
                grads.append(grad.ravel())
        return np.concatenate(grads).astype(np.float64)

    def fun(self, x:np.ndarray,return_grad=True) -> Union[float,Tuple[float,np.ndarray]]:
        """Function to be passed to `scipy.optimize.minimize`,

        :param x: Hyperparameters in 1D representation
        :type x: np.ndarray

        :param return_grad: Return gradients computed via automatic differentiation if 
            `True`. Defaults to `True`.
        :type return_grad: bool, optional

        Returns:
            One of the following, depending on `return_grad`
                - If `return_grad`=`False`, returns only the objective
                - If `return_grad`=`False`, returns a two-element tuple containing:
                    - the objective
                    - a numpy array of the gradients of the objective wrt the 
                      hyperparameters computed via automatic differentiation
        """
        # unpack x and load into module 
        state_dict = self.unpack_parameters(x)
        old_dict = self.model.state_dict()
        old_dict.update(state_dict)
        self.model.load_state_dict(old_dict)
        
        # Getting the related A paramters according to their categorical name
        for i,n in enumerate(self.model.qual_kernel_columns):
            if str(n) in state_dict.keys():
                self.model.A_matrix[i].fci.weight.data = state_dict[str(n)]

        # zero the gradient
        self.model.zero_grad()
        obj = -marginal_log_likelihood(self.model, self.add_prior,self.regularization_parameter) # negative sign to minimize
        
        if return_grad:
            # backprop the objective
            obj.backward()
            
            return obj.item(),self.pack_grads()
        
        return obj.item()


def _sample_from_prior(model) -> np.ndarray:
    out = []
    for _,module,prior,closure,_ in model.named_priors():
        if not closure(module).requires_grad:
            continue
        out.append(prior.expand(closure(module).shape).sample().cpu().numpy().ravel())
    
    return np.concatenate(out)


def cons_f(x,likobj):
    zeta = torch.tensor(likobj.model.zeta, dtype = torch.float64)

    old_dict = likobj.unpack_parameters(x)
    for i,n in enumerate(likobj.qual_kernel_columns):
        if str(n) in old_dict.keys():
            likobj.model.A_matrix[i].fci.weight.data = old_dict[str(n)]

    positions = likobj.model.A_matrix(zeta)
    out_constraint=positions.detach().numpy().reshape(-1,)
    #Omegas=likobj.model.covar_module.base_kernel.kernels[1].lengthscale.data.numpy().reshape(-1,)
    return out_constraint[0:8]

def get_bounds(likobj, theta):

    dic = likobj.unpack_parameters(theta)

    minn = np.empty(0)
    maxx = np.empty(0)
    for name, values in dic.items():
        if len(likobj.model.qual_kernel_columns) > 0:
            if  name == str(likobj.model.qual_kernel_columns[0]):
                minn = np.concatenate( (minn,  np.repeat(-3, values.numel()) ) )
                maxx = np.concatenate( (maxx,  np.repeat(3, values.numel()) ) )
        
        if name == 'likelihood.noise_covar.raw_noise':
            minn = np.concatenate( (minn,  np.repeat(-np.inf, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( np.inf, values.numel()) ) )
        elif name == 'mean_module.constant':
            minn = np.concatenate( (minn,  np.repeat(-np.inf, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( np.inf, values.numel()) ) )
        elif name == 'covar_module.raw_outputscale':
            minn = np.concatenate( (minn,  np.repeat(0, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( np.inf, values.numel()) ) )
        elif 'raw_lengthscale' in name:
            minn = np.concatenate( (minn,  np.repeat(-10.0, values.numel()) ) )
            maxx = np.concatenate( (maxx,  np.repeat( 3.0, values.numel()) ) )
    return np.array(minn).reshape(-1,), np.array(maxx).reshape(-1,)


def _fit_model_from_state(likobj,theta0,jac,options, method = 'trust-constr',constraint=True,bounds=False):
    
    min, max = get_bounds(likobj, theta0)
    bounds_acts = Bounds(min, max)
    
    nonlinear_constraint = NonlinearConstraint(lambda x: cons_f(x, likobj), [0,0,0,0,-5,0,-5,-5],[0,0,5,0,5,5,5,5], jac='2-point', hess=BFGS())
    
    '''
    if constraint==True:
        nonlinear_constraint = NonlinearConstraint(lambda x: cons_f(x, likobj), [0,0,0,0,-inf,0],[0,0,inf,0,inf,inf], jac='2-point', hess=BFGS())
    
    else:
        nonlinear_constraint = NonlinearConstraint(lambda x: cons_f(x, likobj), [-inf,-inf,-inf,-inf,-inf,-inf],[inf,inf,inf,inf,inf,inf], jac='2-point', hess=BFGS())

    '''
    eq_cons = {'type': 'eq',
                'fun' : lambda x: np.array([cons_f(x, likobj)[0],cons_f(x, likobj)[1],cons_f(x, likobj)[3]])}
    ineq_cons = {'type': 'ineq',
                'fun' : lambda x: np.array([cons_f(x, likobj)[2],cons_f(x, likobj)[5]])}
    if constraint==True:
        nonlinear_constraint=[nonlinear_constraint]
    else:
        nonlinear_constraint=[]


    if bounds==True:
        bounds=bounds_acts
    else:
        bounds=None


    try:
        with gptsettings.fast_computations(log_prob=False):
            return minimize(
                fun = likobj.fun,
                x0 = theta0,
                args=(True) if jac else (False),

                method = method,
                jac=jac,
                bounds=bounds,
                constraints= nonlinear_constraint,
                #constraints=[eq_cons, ineq_cons],
                options= options 
            )


    except Exception as e:
        if isinstance(e,NotPSDError) or isinstance(e, NanError):
            # Unstable hyperparameter configuration. This can happen if the 
            # initial starting point is bad. 
            return e
        else:
            # There is some other issue, most likely with the inputs supplied
            # by the user. Raise error to indicate the problematic part.
            raise



def fit_model_scipy(
    model,
    add_prior:bool=True,
    num_restarts:int=24,
    theta0_list:Optional[List[np.ndarray]]=None,
    jac:bool=True, 
    options:Dict={},
    n_jobs:int=-1,
    method = 'L-BFGS-B',
    constraint=False,
    bounds=False,
    regularization_parameter:List[int]=[0,0]
    ) -> Tuple[List[OptimizeResult],float]:
    """Optimize the likelihood/posterior of a GP model using `scipy.optimize.minimize`.

    :param model: A model instance derived from the `models.GPR` class. Can also pass a instance
        inherting from `gpytorch.models.ExactGP` provided that `num_restarts=0` or if all estimable
        parameters/hyperarameters have an associated priors registered to them
    :type model: models.GPR

    :param add_prior: Whether to add the hyperparameter priors to the log-likelihood to optimize the 
        posterior. Optimizing the log-posterior is some what more robust than optimizing the log-likelihood
        when there are few training data. Defaults to True
    :type num_restarts: bool, optional

    :param num_restarts: The number of times to restart the local optimization from a 
        new starting point. This number does not include the initial state of the supplied model. 
        This argument is ignored if initial guesses are explicitly specified. Defaults to 9
    :type num_restarts: int, optional

    :param theta0_list: List of starting points for the hyperparameters. Each list entry should be
        a 1D array. The initial state of the supplied model is ignored. Defaults to None
    :type theta0_list: List[np.ndarray], optional

    :param jac: Use automatic differentiation to compute gradients if `True`. If `False`, uses
        scipy's numerical differentiation mechanism. Defaults to `True`.
    :type jac: bool, optional

    :param options: A dictionary of `L-BFGS-B` options to be passed to `scipy.optimize.minimize`.
        Available options are:
            * ftol: float
                The iteration stops when ``(f^k -
                f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``. Defaults to 1e-6.
            * gtol: float
                The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
                <= gtol`` where ``pg_i`` is the i-th component of the
                projected gradient. Defaults to 1e-5.
            * maxfun: int
                Maximum number of function evaluations. Defaults to 5000
            * maxiter: int 
                Maximum number of iterations. Defaults to 2000
    :type options: dict,optional

    :param n_jobs: Number of jobs to run in parallel. Uses `joblib` for parallelization. Deafults to 1. 
    :type n_jobs: int,optional

    Returns:
        A two-element tuple with the following elements
            * a list of optimization result objects, one for each starting point.
            * the best (negative) log-likelihood/log-posterior found
    
    :rtype: Tuple[List[OptimizeResult],float]
    """

    defaults = {}

    if method == 'L-BFGS-B':
        defaults = {'ftol':1e-6,'gtol':1e-5,'maxfun':5000,'maxiter':2000}
    elif method == 'trust-constr':
        defaults = {'verbose': 1}
    elif method == 'BFGS':
        defaults = {'gtol': 1e-07, 'norm': np.inf, 'eps': 1.4901161193847656e-08,'maxiter': None, 'disp': False, 'return_all': False, 'finite_diff_rel_step': None}#
    elif method == 'SLSQP':
        defaults = { 'maxiter': 100, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}#
    elif method == 'Newton-CG':
        defaults={'xtol': 1e-05, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': False, 'return_all': False}
    else:
        raise ValueError('Wrong method')

    if len(options) > 0:
        for key in options.keys():
            if key not in defaults.keys():
                raise RuntimeError('Unknown option %s!'%key)
            defaults[key] = options[key]


    likobj = MLLObjective(model,add_prior,regularization_parameter)

    if theta0_list is None:
        theta0_list = [likobj.pack_parameters()]
        if num_restarts > -1:
            theta0_list.extend([_sample_from_prior(model) for _ in range(num_restarts+1)])
            theta0_list.pop(0)
    
    # Output - Contains either optimize result objects or exceptions
    set_loky_pickler("dill") 
    out = Parallel(n_jobs=n_jobs,verbose=0)(
        delayed(_fit_model_from_state)(likobj,theta0,jac,defaults, method,constraint,bounds) \
            for theta0 in theta0_list
    )
    set_loky_pickler("pickle")
    
    nlls_opt = [np.inf if isinstance(res,Exception) else res.fun for res in out]
    best_idx = np.argmin(nlls_opt)
    theta_best = out[best_idx].x

    # load best hyperparameters
    old_dict = deepcopy(model.state_dict())
    old_dict.update(likobj.unpack_parameters(theta_best))
    model.load_state_dict(old_dict)

    # if 'fci' in [name for name,p in model.named_parameters()]:
    #     model.nn_model.fci.weight.data = old_dict['fci']

    for i,n in enumerate(model.qual_kernel_columns):
        if str(n) in old_dict.keys():
            model.A_matrix[i].fci.weight.data = old_dict[str(n)]


    return out,nlls_opt[best_idx]