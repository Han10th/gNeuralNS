import torch
import numpy as np
from torch import nn
from utils.utils import differential_y_x,is_leaf
Pout = 0. # kPa
Pext = .9 # 10kPa
class MODEL_NS1D:
    def __init__(self, sampler, N_Q, N_P, N_R,alpha=1,beta=1,rho=1,Kr=1,RRC=[]):
        super(MODEL_NS1D, self).__init__()
        self.sampler = sampler
        self.N_Q = N_Q
        self.N_P = N_P
        self.N_R = N_R
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Kr = Kr
        self.RRC = RRC
    def estimate(self,X,R0=None):
        Q = self.N_Q(X)
        P = self.N_P(X)
        if R0 is None:
            return Q, P
        else:
            R = self.N_R(X) + R0
            A = torch.pi * R * R
            return Q, P, R, A
    def losscompute_all(self,N):
        vesseltree_domain,vesseltree_r0,vesseltree_r0_z = self.sampler.sample_domain(N)
        vesseltree_inlet,inlet_velocity = self.sampler.sample_inlet(N)
        vesseltree_outlet = self.sampler.sample_outlet(N)
        N_segment = len(vesseltree_domain)
        Connect = self.sampler.getConnect()

        N_leaf = 0
        N_branch = 0
        PDEloss1 = 0
        PDEloss2 = 0
        INLETloss = self.DIRICHLETloss(vesseltree_inlet[0],inlet_velocity)
        BRANCHlossQ = 0
        BRANCHlossP = 0
        OUTLETloss = 0
        for j in range(N_segment):
            pdeloss1,pdeloss2 = self.PDEloss(vesseltree_domain[j],
                                             vesseltree_r0[j],
                                             vesseltree_r0_z[j])
            PDEloss1 += pdeloss1
            PDEloss2 += pdeloss2

            idx_list = np.where(Connect[:,j])[0].tolist()
            if len(idx_list) != 0:
                N_branch += 1
                branchlossQ,branchlossP = self.BRANCHloss(vesseltree_outlet[j],vesseltree_inlet,idx_list)
                BRANCHlossQ += branchlossQ
                BRANCHlossP += branchlossP
            if is_leaf(Connect,j):
                N_leaf += 1
                outletloss = self.WINDKESSELloss(vesseltree_outlet[j],self.RRC[j])
                OUTLETloss += outletloss
        return [PDEloss1 / N_segment,PDEloss2 / N_segment,INLETloss,
                BRANCHlossQ / N_branch,BRANCHlossP / N_branch,OUTLETloss / N_leaf]

    def PDEloss(self, X, R0, R0_z):
        Q, P, R, A = self.estimate(X,R0)
        A0 = torch.pi * R0 * R0
        A0_z = 2*torch.pi*R0 * R0_z
        alpha, beta, rho, Kr = self.alpha,self.beta,self.rho,self.Kr

        a01 = beta / (2*torch.sqrt(A)*A0)
        a10 = A/rho - (alpha*2*A0*Q*Q)/(beta*A*torch.sqrt(A))
        a11 = alpha*2*Q / A
        b1 = Kr*Q/A - (alpha*Q*Q)/(A*torch.sqrt(A)) * (2*P/beta + 1/torch.sqrt(A0)) * A0_z

        P_z = differential_y_x(P, X, 0)
        P_t = differential_y_x(A,X,1)
        Q_z = differential_y_x(Q,X,0)
        Q_t = differential_y_x(Q,X,1)

        NS1 = P_t             + a01 * Q_z
        NS2 = Q_t + a10 * P_z + a11 * Q_z + b1
        return MSElosses([NS1]),MSElosses([NS2])
    def DIRICHLETloss(self,X,Q_label):
        Q, P = self.estimate(X)
        return MSElosses([Q - Q_label])
    def BRANCHloss(self, X, X_sub,idx_list):
        N_sub = len(X_sub)
        Q, P = self.estimate(X)
        Q_continuity = Q
        P_continuity = []
        for i in idx_list:
            Qb, Pb = self.estimate(X_sub[i])
            Q_continuity -= Qb
            P_continuity += [P-Pb]
        return MSElosses([Q_continuity]), MSElosses(P_continuity)
    def WINDKESSELloss(self,X,RRC):
        R1, R2, C = RRC[0],RRC[1],RRC[2]
        Q, P = self.estimate(X)
        P_t = differential_y_x(P, X, 1)
        Q_t = differential_y_x(Q, X, 1)
        res = Q * (1 + R1 / R2) + C * R1 * Q_t - (P + Pext - Pout) / R2 * 1e-1 - C * P_t * 1e-1
        return MSElosses([res])

def MSElosses(loss_list):
    Loss = 0
    for loss in loss_list:
        Loss += torch.mean(torch.square(loss))
    return Loss