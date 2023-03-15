from multiprocessing.sharedctypes import Value
import numpy as np
import matplotlib.pyplot as plt
from pyro import param
from scipy.stats.qmc import Sobol, scale
import math
import torch

############################# Bias only #################################


def bias_only(n=10, X=None,
              fidelity=0, noise_std=0.0, random_state=None, shuffle=True):
    """_summary_

    Args: 
        parameters (_type_, optional): For evaluation, you can give parameters and get the values. Defaults to None.
        n (int, optional): defines the number of data needed. Defaults to 100.

    Returns:
        _type_: if paramters are given, it returns y, otherwise it returns both X and y
    """

    if random_state is not None:
        np.random.seed(random_state)

    # input dimension (exclude fidelity level)
    dx = 1

    # upper and lower bound
    l_bound = [-2]
    u_bound = [3]
    out_flag = 0
    # Generate inputs with Sobol sequences if the inputs are not provided
    if X is None:
        sobolset = Sobol(d=dx, seed=random_state)
        # generate a long sequence but only save the first n items
        X = sobolset.random(2 ** (np.log2(n)+1).astype(int))[:n, :]
        # convert a sample from [0,1) to [a,b) with a the lb and b the ub
        X = scale(X, l_bounds=l_bound, u_bounds=u_bound)
        out_flag = 1  # means we generate X inside this function
    if type(X) != np.ndarray:
        X = np.array(X)

    # assign input from Sobol samples
    x = X[..., 0]

    # High fidelity
    if fidelity == 0:
        y = 1/(0.1*np.power(x, 3)+np.power(x, 2)+x+1)
    elif fidelity == 1:
        y = 1/(0.1*np.power(x, 3)+np.power(x, 2)+x+1) - 1
    elif fidelity == 2:
        y = 1/(0.1*np.power(x, 3)+np.power(x, 2)+x+1) + 1
    else:
        raise ValueError("only 3 fidelities of 0,1,2,3 have been implemented")

    # Shuffle
    if X is None:
        if shuffle:
            index = np.random.randint(0, len(y), size=len(y))
            X = X[index, ...]
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


def multi_fidelity_bias_only(X=None,
                             n={"0": 3, "1": 20, "2": 20},
                             noise_std={'0': 0.0, '1': 0.0, '2': 0.0},
                             random_state=None, shuffle=True):
    if X is None:
        X_list = []
        y_list = []
        for level, num in n.items():
            if level in ["0", "1", "2"] and num > 0:
                X, y = bias_only(n=num, fidelity=int(level),
                                 noise_std=noise_std[level], random_state=random_state)
                X = np.hstack([X, np.ones(num).reshape(-1, 1)*float(level)])
                X_list.append(X)
                y_list.append(y)
            else:
                raise ValueError("Wrong label, should be h, l1, or l2")
        return np.vstack([*X_list]), np.hstack(y_list)
    else:
        fidelity = [i for i in n.keys()]
        y_list = []
        if type(X) == np.ndarray:
            X = torch.tensor(X)
        for f in fidelity:
            # Find the index for each fidelity then call the model function
            index = [i[0] for i in torch.argwhere(X[..., -1] == int(f))]
            y_list.append(
                bias_only(X=X[index, 0:-1], fidelity=int(f)), noise_std=noise_std[f])
        return torch.tensor(np.hstack(y_list))
