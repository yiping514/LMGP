import numpy as np
from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
from lmgp_pytorch.utils.input_space import InputSpace

def borehole():
    config = InputSpace()
    r = NumericalVariable(name='r',lower=100,upper=50000)
    Tu = NumericalVariable(name='T_u',lower=63070,upper=115600)
    Hu = NumericalVariable(name='H_u',lower=990,upper=1110)
    Tl = NumericalVariable(name='T_l',lower=63.1,upper=116)
    L = NumericalVariable(name='L',lower=1120,upper=1680)
    K_w = NumericalVariable(name='K_w',lower=9855,upper=12045)
    r_w = CategoricalVariable(name='r_w',levels=np.linspace(0.05,0.15,10))
    H_l = CategoricalVariable(name='H_l',levels=np.linspace(700,820,10))
    config.add_inputs([r,Tu,Hu,Tl,L,K_w,r_w,H_l])
    return config

def piston():
    config = InputSpace()
    M = NumericalVariable(name='M',lower=30,upper=60)
    S = NumericalVariable(name='S',lower=0.005,upper=0.02)
    V0 = NumericalVariable(name='V_0',lower=0.002,upper=0.01)
    Ta = NumericalVariable(name='T_a',lower=290,upper=296)
    T0 = NumericalVariable(name='T_0',lower=340,upper=360)
    k = CategoricalVariable(name='k', levels=np.linspace(1000,5000,10))
    P0 = CategoricalVariable(name='P_0', levels=np.linspace(90000,110000,10))
    config.add_inputs([M,S,V0,Ta,T0,k,P0])
    return config

def rosenbrock():
    config = InputSpace()
    # numerical inputs
    config.add_inputs([
        NumericalVariable(name='x%d'%i, lower=-5, upper=10) for i in [1,2,5,6]
    ])
    config.add_inputs([
        CategoricalVariable(name='x%d'%i, levels=np.linspace(-5,10,10)) for i in [3,4]
    ])
    return config