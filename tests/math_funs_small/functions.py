import math
from typing import Dict

def branin(params:Dict)-> float:
    x = [params['x%d'%(i+1)] for i in range(len(params))]
    term1 = x[1] - 5.1 / (4 * math.pi ** 2) * x[0] ** 2 + 5 / math.pi * x[0] - 6
    term2 = 10 * (1 - 1 / (8 * math.pi)) * math.cos(x[0]) + 10
    return term1**2 + term2

def camel(params:Dict)-> float:
    x1 = params['x1'];x2 = params['x2']
    term1 = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 
    term2 = x1 * x2 + (4 * x2 ** 2 - 4) * x2 ** 2
    return term1 + term2
