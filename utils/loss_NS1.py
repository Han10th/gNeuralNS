import torch
import numpy as np
from torch import nn
from utils.utils import differential_y_x,is_leaf
Pout = 0. # kPa
Pext = .9 # 10kPa
class MODEL_NS1D:
    def __init__(self, sampler, N_Q_list, N_P_list, N_R_list,alpha=1,beta=1,rho=1,Kr=1,RRC=[]):
        super(MODEL_NS1D, self).__init__()
        self.sampler = sampler
        self.N_Q_list = N_Q_list
        self.N_P_list = N_P_list
        self.N_R_list = N_R_list
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Kr = Kr
        self.RRC = RRC
    def estimate(self,X,N_Q,N_P,N_R=None,R0=None):
        Q = N_Q(X)
        P = N_P(X)
        if R0 is None:
            return Q, P
        else:
            R = N_R(X) + R0
            A = torch.pi * R * R
            return Q, P, R, A
    def losscompute_all(self,N):
        time = self.sampler.sample_time(N)
        vesseltree_domain,vesseltree_r0,vesseltree_r0_z = self.sampler.sample_domain(N,time)
        vesseltree_inlet,inlet_velocity = self.sampler.sample_inlet(N,time)
        vesseltree_outlet = self.sampler.sample_outlet(N,time)
        N_segment = len(vesseltree_domain)
        Connect = self.sampler.getConnect()

        N_leaf = 0
        N_branch = 0
        PDEloss1 = 0
        PDEloss2 = 0
        INLETloss = self.DIRICHLETloss(vesseltree_inlet[0],inlet_velocity,0)
        BRANCHlossQ = 0
        BRANCHlossP = 0
        OUTLETloss = 0
        for j in range(N_segment):
            pdeloss1,pdeloss2 = self.PDEloss(vesseltree_domain[j],
                                             vesseltree_r0[j],
                                             vesseltree_r0_z[j],
                                             j)
            PDEloss1 += pdeloss1
            PDEloss2 += pdeloss2

            idx_list = np.where(Connect[:,j])[0].tolist()
            if len(idx_list) != 0:
                N_branch += 1
                branchlossQ,branchlossP = self.BRANCHloss(vesseltree_outlet[j],vesseltree_inlet,j,idx_list)
                BRANCHlossQ += branchlossQ
                BRANCHlossP += branchlossP
            if is_leaf(Connect,j):
                N_leaf += 1
                outletloss = self.WINDKESSELloss(vesseltree_outlet[j],self.RRC[j],j)
                OUTLETloss += outletloss
        N_branch = 1 if N_branch==0 else N_branch
        return [PDEloss1 / N_segment,PDEloss2 / N_segment,INLETloss / 1,
                BRANCHlossQ / N_branch,BRANCHlossP / N_branch,OUTLETloss / N_leaf]

    def PDEloss(self, X, R0, R0_z, j):
        Q, P, R, A = self.estimate(X,
                                   self.N_Q_list[j],
                                   self.N_P_list[j],
                                   self.N_R_list[j],
                                   R0)
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
    def DIRICHLETloss(self,X,Q_label,j):
        Q, P = self.estimate(X,self.N_Q_list[j],self.N_P_list[j],self.N_R_list[j])
        return MSElosses([Q - Q_label])
    def BRANCHloss(self, X, X_sub, j, idx_list):
        Q, P = self.estimate(X,self.N_Q_list[j],self.N_P_list[j],self.N_R_list[j])
        Q_branchlist = []
        P_continuity = []
        for i in idx_list:
            Qb, Pb = self.estimate(X_sub[i],self.N_Q_list[i],self.N_P_list[i],self.N_R_list[i])
            Q_branchlist += [Qb]
            P_continuity += [P-Pb]
        Q_continuity = [Q - torch.concat(Q_branchlist,axis=1).sum(axis=1,keepdim=True)]
        return MSElosses(Q_continuity), MSElosses(P_continuity)
    def WINDKESSELloss(self,X,RRC,j):
        R1, R2, C = RRC[0],RRC[1],RRC[2]
        Q, P = self.estimate(X, self.N_Q_list[j], self.N_P_list[j], self.N_R_list[j])
        P_t = differential_y_x(P, X, 1)
        Q_t = differential_y_x(Q, X, 1)
        res = Q * (1 + R1 / R2) + C * R1 * Q_t - (P + Pext - Pout) / R2 * 1e-1 - C * P_t * 1e-1
        return MSElosses([res])

# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8), dpi=150)
# plt.subplot(2,1,1)
# plt.plot(X[:,1].detach().cpu().numpy(),
#          Q[:,0].detach().cpu().numpy(),'r.')
# plt.plot(X_sub[i][:,1].detach().cpu().numpy(),
#          Qb[:,0].detach().cpu().numpy(),'b.')
# plt.subplot(2,1,2)
# plt.plot(X[:,1].detach().cpu().numpy(),
#          Q_continuity[0][:,0].detach().cpu().numpy(),'g.')
# plt.plot(X[:,1].detach().cpu().numpy(),
#          (torch.abs(Q[:,0]-Qb[:,0])).detach().cpu().numpy(),'y.')
# plt.show()

def MSElosses(loss_list):
    Loss = 0
    N = len(loss_list)
    for loss in loss_list:
        Loss += torch.mean(torch.square(loss))
    return Loss/N