import torch
import numpy as np
from torch import nn
from utils.utils import differential_y_x
class MODEL_FSI:
    def __init__(self, N_u, N_p, N_s=None, Re=1,cof_a=1,cof_H=1):
        super(MODEL_FSI, self).__init__()
        self.N_u = N_u
        self.N_p = N_p
        self.N_s = N_s
        self.Re = Re
        self.cof_a = cof_a
        self.cof_H = cof_H


    def losscompute(self, X,
                   label=None,
                   normal=None,
                   LOSS_TYPE = 'PDE'):
        if self.N_s is None:
            Y = X
            U = torch.concat((
                self.N_u(Y), self.N_p(Y)
            ), axis=1)
        else:
            D = self.N_s(X)
            Y = torch.concat((X[:,0:1]+D[:,0:1],X[:,1:2]+D[:,1:2],X[:,-1:]),axis=1)
            U = torch.concat((
                self.N_u(Y), self.N_p(Y)
            ), axis=1)

        if LOSS_TYPE == 'PDE':
            return ns_pde(Y,U, self.Re)
        elif LOSS_TYPE == 'Dirichlet':
            return Lf_dirichlet(U,label)
        elif LOSS_TYPE == 'Neutral':
            return Lf_neutral(Y, U, normal, self.Re)
        elif LOSS_TYPE == 'Windkssel':
            print('WARNING: To be implemented!')
        elif LOSS_TYPE == 'StressContinuity':
            return Li_stress(Y,U,X,D,normal,self.cof_a,self.cof_H)
        else:
            print('WARNING: Please state the loss wanted!')


def ns_pde(Y, U, Re=1):
    u_vel, v_vel, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]

    u_vel_x = differential_y_x(u_vel,Y,0)
    u_vel_y = differential_y_x(u_vel,Y,1)
    u_vel_t = differential_y_x(u_vel,Y,2)
    u_vel_xx = differential_y_x(u_vel_x,Y,0)
    u_vel_yy = differential_y_x(u_vel_y,Y,1)

    v_vel_x = differential_y_x(v_vel,Y,0)
    v_vel_y = differential_y_x(v_vel,Y,1)
    v_vel_t = differential_y_x(v_vel,Y,2)
    v_vel_xx = differential_y_x(v_vel_x,Y,0)
    v_vel_yy = differential_y_x(v_vel_y,Y,1)

    p_x = differential_y_x(p,Y,0)
    p_y = differential_y_x(p,Y,1)

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
    dirichlet_x = ((u_vel - u_label[:, 0:1]))
    dirichlet_y = ((v_vel - u_label[:, 1:2]))
    return [dirichlet_x,dirichlet_y]

def Lf_neutral(Y, U, n, Re=1):
    u_vel, v_vel, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]

    u_vel_x = differential_y_x(u_vel,Y,0)
    u_vel_y = differential_y_x(u_vel,Y,1)
    v_vel_x = differential_y_x(v_vel,Y,0)
    v_vel_y = differential_y_x(v_vel,Y,1)

    neutral_x = (
            -p * 1 + 1 / Re * (u_vel_x * 1 + u_vel_y * 0))
    neutral_y = (
            -p * 0 + 1 / Re * (v_vel_x * 1 + v_vel_y * 0))
    return [neutral_x, neutral_y]

def Li_stress(Y,U,X,D,normal,cof_a,cof_H):
    # dx, dy = D[:, 0:1], D[:, 1:2]
    dy = D
    u_vel, v_vel, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]

    dy_t = differential_y_x(dy, X, 2)
    dy_tt = differential_y_x(dy_t, X, 2)

    # u_vel_x = differential_y_x(u_vel,Y,0)
    # u_vel_y = differential_y_x(u_vel,Y,1)
    # v_vel_x = differential_y_x(v_vel,Y,0)
    # v_vel_y = differential_y_x(v_vel,Y,1)
    # Du = torch.concat((
    #     torch.concat((u_vel_x.unsqueeze(-1), u_vel_y.unsqueeze(-1)), dim=2),
    #     torch.concat((v_vel_x.unsqueeze(-1), v_vel_y.unsqueeze(-1)), dim=2)), dim=1)
    # SigF = mu_f * (Du)
    # stressF = torch.sum((p) * normal[:, 1:2], dim=1) - torch.matmul(SigF, normal)

    stressF = (p - 0)/ cof_H
    stressS = dy_tt + cof_a * dy
    l_FSstr = torch.square(stressS - stressF)
    return [l_FSstr]


# x = np.array([[2,3]])
# X = torch.autograd.Variable(torch.from_numpy(x).float(),
#      requires_grad=True)
# D = X[:,0:1]*X[:,1:2]
# Y=torch.concat((X[:,0:1]+D,X[:,1:2]),axis=1)
# F=Y
# torch.autograd.grad(F.sum(), X)[0]
# torch.autograd.grad(F.sum(), Y)[0]