import numpy as np
import torch
from utils.networks import FNN
from utils.loss_fluid2 import MODEL_FLUID
from utils.sampler_rectangle2 import SAMPLER_RECTANGLE
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
domain = [1,1]
time = [0,1,2]
visgrid=[6,3,2]
Re = 1
Num = 1000
lr_u = 1e-3
lr_p = 1e-3

N_u = FNN([3] + 4 * [40] + [2], "tanh", "Glorot normal").to(device)
N_p = FNN([3] + 4 * [20] + [1], "tanh", "Glorot normal").to(device)
Opt_u = torch.optim.LBFGS(list(N_u.parameters()), lr=lr_u)
Opt_p = torch.optim.LBFGS(list(N_p.parameters()), lr=lr_p)
Model_fluid = MODEL_FLUID(N_u,N_p,Re)

sampler = SAMPLER_RECTANGLE(domain, time, visgrid,device=device)
Xdomain,NboundR,XboundL,XboundR,XboundU,XboundD,VboundL,Vbound0 = sampler(Num)

Ldomain = Model_fluid(Xdomain,LOSS_TYPE = 'PDE')
LboundL = Model_fluid(XboundL,label = VboundL,LOSS_TYPE = 'DIRICHLET')
LboundR = Model_fluid(XboundR,normal = NboundR,LOSS_TYPE = 'NEUTRAL')
LboundU = Model_fluid(XboundU,label = Vbound0,LOSS_TYPE = 'DIRICHLET')
LboundD = Model_fluid(XboundD,label = Vbound0,LOSS_TYPE = 'DIRICHLET')

Opt_u.zero_grad()
Opt_p.zero_grad()
alphas = [1,100,100,100,100]
losses = [Ldomain,LboundL,LboundR,LboundU,LboundD]
N_loss = len(alphas)
Loss_total = 0
for i in range(N_loss):
    for loss in losses[i]:
        Loss_total += alphas[i]*torch.mean(torch.square(loss))
Loss_total.backward()
# Opt_u.step()
# Opt_p.step()

sampler.visualize(N_u,N_p)