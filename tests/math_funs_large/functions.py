import math
import numpy as np
from typing import Dict

def rosenbrock(params:Dict)->float:
    x = np.array([params['x%d'%(i+1)] for i in range(len(params))])
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2)