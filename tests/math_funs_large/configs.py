import numpy as np
from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
from lmgp_pytorch.utils.input_space import InputSpace

def rosenbrock():
    config = InputSpace()
    # numerical inputs
    config.add_inputs([
        NumericalVariable(name='x%d'%i, lower=-5, upper=10) for i in [1,3,5,7,9]
    ])
    config.add_inputs([
        CategoricalVariable(name='x%d'%i, levels=np.linspace(-5,10,10)) for i in [2,4,6,8,10]
    ])
    return config