from random import sample
import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np
from numpy import sin, cos
from pyDOE import lhs
import scipy.io
import scipy.integrate as integrate


torch.manual_seed(123)
np.random.seed(123)

def derivs(state, t, g, L, M, D, tau):

    dxdt = np.zeros_like(state)
    ##################################
    #状態方程式
    # 角度 a = steta[0], 角速度 w = steta[1]
    # dadt = w, dwdt = ...
    dxdt[0] = state[1]
    dxdt[1] = -D/(M*L)*dxdt[0] -g/L*np.sin(state[0]) + tau/(M*L**2)
    ##################################

    return dxdt

def func(x, tau, L, m, d, g):
    
    dxdt = np.concatenate(([x[0,[1]]], [-g/L*np.sin(x[0,[0]])-d/(m*L)*x[0,[1]]+1/(m*L**2)*tau]), axis=1)

    return dxdt

def solve_ode(x, tau, g, L, d, m, delta_t, t):
    sol = np.zeros((int(t/delta_t), 2))
    for i in range(int(t/delta_t)):
        
        f1 = func(x, tau[i], L, d, m, g)
        f2 = func(x + f1*delta_t/2, tau[i], L, d, m, g)
        f3 = func(x + f2*delta_t/2, tau[i], L, d, m, g)
        f4 = func(x + f3*delta_t, tau[i], L, d, m, g)

        x = x + (f1 + 2*f2 + 2*f3 + f4)*delta_t/6
        # print(x_hat)

        sol[i, :] = x

    return sol