import torch
import numpy as np
from torch import nn
from utils.utils import differential_y_x,is_leaf
Pout = 0. # kPa
Pext = 0 # 10kPa
class MODEL_NS1D:
    def __init__(self, sampler, N_Q_list, N_P_list,alpha=1,beta=1,rho=1,Kr=1,P_ext=0,RRC=[]):
        super(MODEL_NS1D, self).__init__()
        self.sampler = sampler
        self.N_Q_list = N_Q_list
        self.N_P_list = N_P_list
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Kr = Kr
        self.P_ext = P_ext
        self.RRC = RRC
    def estimate(self,X,N_Q,N_P):
        Q = N_Q(X)
        P = N_P(X)
        return Q, P
    def losscompute_demo(self,N):
        time = self.sampler.sample_time(N)
        vesseltree_demo,vesseltree_velocity = self.sampler.sample_demo(N,time)
        N_segment = len(vesseltree_demo)

        LABELloss = 0
        for j in range(N_segment):
            LABELloss += self.DIRICHLETloss(vesseltree_demo[j],vesseltree_velocity[j],j)
        return [0,0,LABELloss,0,0,0]

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
        # N_segment = 1
        for j in range(N_segment):
            pdeloss1,pdeloss2 = self.PDEloss(vesseltree_domain[j],
                                             vesseltree_r0[j],
                                             vesseltree_r0_z[j],
                                             j)
            a1 = 1 if j == 0 else 100
            PDEloss1 += a1*pdeloss1
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
        N_leaf = 1 if N_leaf == 0 else N_leaf
        return [PDEloss1 / N_segment,PDEloss2 / N_segment,INLETloss / 1,
                BRANCHlossQ / N_branch,BRANCHlossP / N_branch,OUTLETloss / N_leaf]

    def PDEloss(self, X, R0, R0_z, j):
        Q, P = self.estimate(X,self.N_Q_list[j],self.N_P_list[j])
        A0 = torch.pi * R0 * R0
        A0_z = 2*torch.pi*R0 * R0_z
        alpha, beta, rho, Kr, P_ext = self.alpha,self.beta,self.rho,self.Kr,self.P_ext

        aP = (A0/beta) * (P-P_ext) + torch.sqrt(A0)
        a01 = (beta) / (2*aP*A0)
        a10 = (aP*aP) / (rho) - (2*A0*Q*Q)/(beta*aP*aP*aP)
        a11 = (2*alpha*Q) / (aP*aP)
        b1 = Kr*Q/(aP*aP) - (alpha*Q*Q*A0_z)/(aP*aP*aP) * ((2/beta)*(P-P_ext) + 1/torch.sqrt(A0))

        P_z = differential_y_x(P,X,0)
        P_t = differential_y_x(P,X,1)
        Q_z = differential_y_x(Q,X,0)
        Q_t = differential_y_x(Q,X,1)

        NS1 = P_t             + a01 * Q_z
        NS2 = Q_t + a10 * P_z + a11 * Q_z + b1
        return MSElosses([NS1]),MSElosses([NS2])
    def DIRICHLETloss(self,X,Q_label,j):
        Q, P = self.estimate(X,self.N_Q_list[j],self.N_P_list[j])
        numj = 1 if j == 0 else 2
        return MSElosses([Q - Q_label/numj]) + MSElosses([P/10 - Q_label/(np.sqrt(numj))])
    def BRANCHloss(self, X, X_sub, j, idx_list):
        Q, P = self.estimate(X,self.N_Q_list[j],self.N_P_list[j])
        Q_branchlist = []
        P_continuity = []
        for i in idx_list:
            Qb, Pb = self.estimate(X_sub[i],self.N_Q_list[i],self.N_P_list[i])
            Q_branchlist += [Q-Qb]
            P_continuity += [P-Pb]
        Q_continuity = [Q - torch.concat(Q_branchlist,axis=1).sum(axis=1,keepdim=True)]
        return MSElosses(Q_branchlist), MSElosses(P_continuity)
    def WINDKESSELloss(self,X,RRC,j):
        R1, R2, C = RRC[0],RRC[1],RRC[2]
        Q, P = self.estimate(X, self.N_Q_list[j], self.N_P_list[j])
        P_t = differential_y_x(P, X, 1)
        Q_t = differential_y_x(Q, X, 1)
        res = Q * (1 + R1 / R2) + C * R1 * Q_t - P / R2 - C * P_t
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
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 8), dpi=150)
# plt.subplot(2,1,1)
# plt.plot(X[:,1].detach().cpu().numpy(),
#          P[:,0].detach().cpu().numpy(),'r.')
# plt.plot(X_sub[i][:,1].detach().cpu().numpy(),
#          Pb[:,0].detach().cpu().numpy()+5,'b.')
# plt.subplot(2,1,2)
# plt.plot(X[:,1].detach().cpu().numpy(),
#          (P-Pb)[:,0].detach().cpu().numpy(),'g.')
# plt.show()

def MSElosses(loss_list):
    Loss = 0
    N = len(loss_list)
    for loss in loss_list:
        Loss += torch.mean(torch.square(loss))
    return Loss/N