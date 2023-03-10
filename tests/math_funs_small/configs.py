import numpy as np
from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
from lmgp_pytorch.utils.input_space import InputSpace

def branin():
    config = InputSpace()
    config.add_input(NumericalVariable('x1', -5, 10))
    config.add_input(CategoricalVariable(name='x2', levels=np.linspace(0.,15.,5)))
    return config

def camel():
    config = InputSpace()
    config.add_input(NumericalVariable('x1', -3, 3))
    config.add_input(CategoricalVariable(name='x2', levels=np.linspace(-2,2,5)))
    return config