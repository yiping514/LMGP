import math
import numpy as np
from typing import Dict

def borehole(params:Dict)->float:
    numerator = 2*math.pi*params['T_u']*(params['H_u']-params['H_l'])
    den_term1 = math.log(params['r']/params['r_w'])
    den_term2 = 1+ 2*params['L']*params['T_u']/(den_term1*params['r_w']**2*params['K_w']) + \
        params['T_u']/params['T_l']
    
    return numerator/den_term1/den_term2

def piston(params:Dict)->float:
    A = params['P_0']*params['S'] + 19.62*params['M'] - params['k']*params['V_0']/params['S']
    term1 = params['P_0']*params['V_0']/params['T_0']*params['T_a']
    V = params['S']/2/params['k']*(math.sqrt(A**2 + 4*term1)-A)
    term2 = params['k'] + (params['S']**2)*term1/(V**2)
    return 2*math.pi*math.sqrt(params['M']/term2)

def rosenbrock(params:Dict)->float:
    x = np.array([params['x%d'%(i+1)] for i in range(len(params))])
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2)