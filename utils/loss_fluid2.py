import torch
import numpy as np
from torch import nn

class MODEL_FLUID(nn.Module):
    def __init__(self, N_u, N_p, Re):
        super(MODEL_FLUID, self).__init__()
        self.N_u = N_u
        self.N_p = N_p
        self.Re = Re

    def forward(self, X,
                   label=None,
                   normal=None,
                   LOSS_TYPE = 'PDE'):
        U = torch.concat((
            self.N_u(X), self.N_p(X)
        ), axis=1)

        if LOSS_TYPE == 'PDE':
            return ns_pde(X,U, self.Re)
        elif LOSS_TYPE == 'DIRICHLET':
            return Lf_dirichlet(U,label)
        elif LOSS_TYPE == 'NEUTRAL':
            return Lf_neutral(X, U, normal, self.Re)
        elif LOSS_TYPE == 'WINKSELL':
            print('WARNING: To be implemented!')
        else:
            print('WARNING: Please state the loss wanted!')




def differential_y_x(y,X,component):
    return torch.autograd.grad(y.sum(), X, create_graph=True)[0][:,component:component+1]
'''The losses for fluid problem'''
def ns_pde(X, U, Re=1):
    u_vel, v_vel, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    x_cor, y_cor, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]

    u_vel_X = torch.autograd.grad(u_vel.sum(), X, create_graph=True)[0]
    u_vel_x = differential_y_x(u_vel,X,0)
    u_vel_y = differential_y_x(u_vel,X,1)
    u_vel_t = differential_y_x(u_vel,X,2)
    u_vel_xx = differential_y_x(u_vel_x,X,0)
    u_vel_yy = differential_y_x(u_vel_y,X,1)

    v_vel_x = differential_y_x(v_vel,X,0)
    v_vel_y = differential_y_x(v_vel,X,1)
    v_vel_t = differential_y_x(v_vel,X,2)
    v_vel_xx = differential_y_x(v_vel_x,X,0)
    v_vel_yy = differential_y_x(v_vel_y,X,1)

    p_x = differential_y_x(p,X,0)
    p_y = differential_y_x(p,X,1)

    momentum_x = (u_vel_t
                  + (u_vel * u_vel_x + v_vel * u_vel_y)
                  + p_x - 1 / Re * (u_vel_xx + u_vel_yy))
    momentum_y = (v_vel_t
                  + (u_vel * v_vel_x + v_vel * v_vel_y)
                  + p_y - 1 / Re * (v_vel_xx + v_vel_yy))
    continuity = u_vel_x + v_vel_y
    return [momentum_x, momentum_y, continuity]

def Lf_dirichlet(u, u_label):
    u_vel, v_vel, p = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    dirichlet_x = (torch.square(u_vel - u_label[:, 0:1]))
    dirichlet_y = (torch.square(v_vel - u_label[:, 1:2]))
    return [dirichlet_x,dirichlet_y]

def Lf_neutral(X, U, n, Re=1):
    u_vel, v_vel, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]

    u_vel_x = differential_y_x(u_vel,X,0)
    u_vel_y = differential_y_x(u_vel,X,1)
    v_vel_x = differential_y_x(v_vel,X,0)
    v_vel_y = differential_y_x(v_vel,X,1)

    neutral_x = (
            -p * 1 + 1 / Re * (u_vel_x * 1 + u_vel_y * 0))
    neutral_y = (
            -p * 0 + 1 / Re * (v_vel_x * 1 + v_vel_y * 0))
    return [neutral_x, neutral_y]