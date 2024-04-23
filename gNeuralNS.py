import numpy as np
import torch
from utils.networks import FNN
from utils.domain2 import ELLIPSE
from utils.loss_fluid2 import MODEL_FLUID
from utils.sampler_rectangle2 import SAMPLER_RECTANGLE
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Region definition
domain = [3.0,1.0]
time = [0,1/2,2]
# Object definition
ellipse_center_ref=[1.0,0.0]
ellipse_size=[0.1,0.05]
ellipse_period = [1,1]
ellipse_range=[0.0,0.3]
# Fluid Parameter
Re = 40
# Networks Parameter
epoch_N = 100000
sample_N = 1000
lr_u = 1e-3
lr_p = 1e-3
# Visualization Parameter
visgrid=[40,20,5]

N_u = FNN([3] + 10 * [40] + [2], "tanh", "Glorot normal").to(device)
N_p = FNN([3] + 10 * [20] + [1], "tanh", "Glorot normal").to(device)
Opt_u = torch.optim.Adam(list(N_u.parameters()), lr=lr_u)
Opt_p = torch.optim.Adam(list(N_p.parameters()), lr=lr_p)
Model_fluid = MODEL_FLUID(N_u,N_p,Re)

ellipse = ELLIPSE(ellipse_center_ref, ellipse_size, ellipse_period, ellipse_range,time)
sampler = SAMPLER_RECTANGLE(domain, time, visgrid,device=device, func_object=ellipse)
ellipse.visualize_sampled_domain(domain)

for epoch_i in range(epoch_N):
    Xdomain,XbdrDiri1,UbdrDiri1,XbdrDiri0,UbdrDiri0,XbdrNeutral,NbdrNeutral = sampler(sample_N)
    
    Ldomain = Model_fluid(Xdomain,LOSS_TYPE = 'PDE')
    LbdrDiri1 = Model_fluid(XbdrDiri1,label = UbdrDiri1,LOSS_TYPE = 'Dirichlet')
    LbdrDiri0 = Model_fluid(XbdrDiri0,label = UbdrDiri0,LOSS_TYPE = 'Dirichlet')
    LbdrNeutral = Model_fluid(XbdrNeutral,normal = NbdrNeutral,LOSS_TYPE = 'Neutral')
    
    Opt_u.zero_grad()
    Opt_p.zero_grad()
    alphas = [1e-5,1,1,1]
    loss_list = [Ldomain,LbdrDiri1,LbdrDiri0,LbdrNeutral]
    Losses = []
    N_loss = len(alphas)
    Loss_total = 0
    for i in range(N_loss):
        Loss = 0
        for loss in loss_list[i]:
            Loss += torch.mean(torch.square(loss))
        Losses += [Loss]
        Loss_total += alphas[i]*Losses[i]
    Loss_total.backward()
    Opt_u.step()
    Opt_p.step()


    if epoch_i%100==0:
        print("EPOCH : {:5d} \t Total: {:5.8f} \t PDE: {:5.8f} \t Inlet: {:5.8f} \t BdrDiri: {:5.8f} \t Outlet: {:5.8f}".format(
            epoch_i,Loss_total, Losses[0], Losses[1], Losses[2], Losses[3]))

    if epoch_i%2000==0:
        sampler.visualize(N_u,N_p)