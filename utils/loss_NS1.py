import torch
import numpy as np
from torch import nn
from utils.utils import differential_y_x
class MODEL_FSI:
    def __init__(self, N_Q, N_P, N_A, alpha=1,beta=1,rho=1,Kr=1):
        super(MODEL_FSI, self).__init__()
        self.N_Q = N_Q
        self.N_P = N_P
        self.N_A = N_A
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Kr = Kr

    def losscompute(self, X,
                   label=None,
                   normal=None,
                   LOSS_TYPE = 'PDE'):

        Q = self.N_Q(X)
        P = self.N_P(X)
        A = self.N_A(X)

        if LOSS_TYPE == 'PDE':
            return ns1d_pde(X,Q,P,A,
                            self.alpha,
                            self.beta,
                            self.rho,
                            self.Kr)
        else:
            print('WARNING: Please state the loss wanted!')

def ns1d_pde(X,Q,P,A,alpha,beta,rho,Kr):
    A_t = differential_y_x(A,X,1)

    Q_z = differential_y_x(Q,X,0)

    Q_t = differential_y_x(Q,X,1)

    Q2A_z = differential_y_x(Q*Q/A,X,0)

    P_z = differential_y_x(P,X,0)