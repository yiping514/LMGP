from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from pyro import param
from scipy.stats.qmc import Sobol, scale
import math
import torch

################################## Wing #########################################
def wing(n=100, X = None, 
    fidelity = 0,noise_std = 0.0, random_state = None, shuffle = True):
    """_summary_

    Args:
        parameters (_type_, optional): For evaluation, you can give parameters and get the values. Defaults to None.
        n (int, optional): defines the number of data needed. Defaults to 100.

    Returns:
        _type_: if paramters are given, it returns y, otherwise it returns both X and y
    """

    if random_state is not None:
        np.random.seed(random_state)

    dx = 10
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    out_flag = 0
    if X is None:
        sobolset = Sobol(d=dx, seed = random_state)
        X = sobolset.random(2 ** (np.log2(n) + 1).astype(int))[:n, :]
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1
    if type(X) != np.ndarray:
        X = np.array(X)
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (np.pi/180.0)
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    # This is the output

    if fidelity == 0:
        y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
            q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3) *\
            (Nz * Wdg) ** 0.49 + Sw * Wp
    elif fidelity == 1:
    # This is the output
        y = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
            q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
            * (Nz * Wdg) ** 0.49 + 1 * Wp 

    elif fidelity == 2:
    # This is level 2 in Tammers paper
        y = 0.036 * Sw**0.8 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
            q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
            * (Nz * Wdg) ** 0.49 + 1 * Wp


    elif fidelity == 3:
    # This is level 2 in Tammers paper
        y = 0.036 * Sw**0.9 * Wfw**0.0035 * (A/(np.cos(Gama)) ** 2) ** 0.6 * \
            q**0.006*lamb**0.04 * ((100 * tc)/(np.cos(Gama)))**(-0.3)\
            * (Nz * Wdg) ** 0.49 + 0 * Wp
    else:
        raise ValueError('only 4 fidelities of 0,1,2,3 have been implemented ')

    if X is None:
        if shuffle:
            index = np.random.randint(0, len(y), size = len(y))
            X = X[index,...]
            y = y[index]


    if noise_std > 0.0:
        if out_flag == 1:
            return X, y + np.random.randn(*y.shape) * noise_std
        else:
            return y       
    else:
        if out_flag == 1:
            return X, y
        else:
            return y

################################# Multi-fidelity ####################################
def multi_fidelity_wing(X = None,
    n={'0': 50, '1': 100, '2': 100, '3': 100},
    noise_std={'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0},
    random_state = None, shuffle = True):
    if X is None:
        X_list = []
        y_list = []
        for level, num in n.items():
            if level  in ['0','1','2','3'] and num > 0:
                X, y = wing(n=num, fidelity = int(level), 
                    noise_std= noise_std[level], random_state = random_state)
                X = np.hstack([X, np.ones(num).reshape(-1, 1) * float(level)])
                X_list.append(X)
                y_list.append(y)
            else:
                raise ValueError('Wrong label, should be h, l1, l2 or l3')
        return np.vstack([*X_list]), np.hstack(y_list)
    else:
        fidelity = [i for i in n.keys()]
        y_list = []
        if type(X) == np.ndarray:
            X = torch.tensor(X)
        for f in fidelity:
            # Find the index for each fidelity then call the wing
            index = [i[0] for i in torch.argwhere(X[...,-1]== int(f))]
            y_list.append(wing(X=X[index, 0:-1], fidelity=int(f),  
                noise_std= noise_std[f]))
        return torch.tensor(np.hstack(y_list))



def multi_fidelity_wing_value(input):
    # if type(input) == torch.Tensor:
    #     input_copy = input.clone()
    # elif type(input) == np.ndarray:
    #     input_copy = np.copy(input)
    y_list = []
    for value in input:
        if value[-1] == 0.0:
            y_list.append(wing(X=value))
        elif value[-1] == 1.0:
            y_list.append(wing(X=value))
        elif value[-1] == 2.0:
            y_list.append(wing(X=value))
        elif value[-1] == 3.0:
            y_list.append(wing(X=value))
        else:
            raise ValueError('Wrong label, should be 0, 1, 2 or 3')
    return torch.tensor(np.hstack(y_list))


def Augmented_branin(input, negate = True, mapping = None):

    X = input.clone()

    if mapping is not None:
        X[..., 2] = torch.tensor(list(map(lambda x: mapping[str(float(x))], X[..., 2]))).to(X)

    t1 = (
        X[..., 1]
        - (5.1 / (4 * math.pi ** 2) - 0.1 * (1 - X[..., 2])) * X[..., 0] ** 2
        + 5 / math.pi * X[..., 0]
        - 6
    )
    t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
    return -(t1 ** 2 + t2 + 10) if negate else (t1 ** 2 + t2 + 10)
