import numpy as np
import torch
from utils.utils import check_folder,save_cpt,load_cpt
from utils.networks import FNN
from utils.domain2 import ELLIPSE
from utils.loss_fsi2 import MODEL_FSI
from utils.sampler_rectangle2 import SAMPLER_RECTANGLE
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# File io
folder_ckp = 'node/disk_problem/'
check_folder(folder_ckp)
# Visualization Parameter
# visgrid=[80,40,80]
visgrid=[40,20,5]
# Region definition
domain = [3.0,1.0]
time = [-0.2,0,1/2,2+0.2]
# time = [0,0,2,2]
# Seq2Seq parameter
sequence_N = 50
time_step = (time[-1]+ time[0]) / sequence_N
time_bound = time[0] + time_step
# Object definition
ellipse_center_ref=[1.0,0.0]
ellipse_size=[0.1,0.05]
ellipse_period = [1,1]
ellipse_range=[0.0,0.3]
# Fluid Parameter
Re = 100
# Networks Parameter
epoch_N = 100000
sample_N = 1000
lr_u = 1e-3
lr_p = 1e-3
a_ns = 1e-7
alphas = [a_ns,10,1,1,1,1]
# Alpha_ns dynamic weighting
N_loss = len(alphas)
this_N = 0

N_u = FNN([3] + 10 * [40] + [2], "tanh", "Glorot normal").to(device)
N_p = FNN([3] + 10 * [20] + [1], "tanh", "Glorot normal").to(device)
# load_cpt(folder_ckp+'{:d}'.format(30000),N_u,N_p)
load_cpt(folder_ckp+'best',N_u,N_p)
Opt_u = torch.optim.Adam(list(N_u.parameters()), lr=lr_u)
Opt_p = torch.optim.Adam(list(N_p.parameters()), lr=lr_p)
model_fsi = MODEL_FSI(N_u,N_p,None,Re)

ellipse = ELLIPSE(ellipse_center_ref, ellipse_size, ellipse_period, ellipse_range,time)
sampler = SAMPLER_RECTANGLE(domain, time, visgrid,device=device, func_object=ellipse)
sampler.update_time(time_bound)
# ellipse.visualize_sampled_domain(domain)

for epoch_i in range(epoch_N):
    this_N = this_N + 1
    Xdomain = sampler.sample_domain(sample_N)
    XbdrIN,     UbdrIN,     NbdrIN  = sampler.sample_inlet(sample_N)
    XbdrWALL,   UbdrWALL,   NbdrWALL= sampler.sample_wall(sample_N)
    XbdrOUT,    UbdrOUT,    NbdrOUT = sampler.sample_outlet(sample_N)
    Xobject,    Uobject = sampler.sample_objects(sample_N)

    Ldomain = model_fsi.losscompute(Xdomain,LOSS_TYPE = 'PDE')
    LbdrIN = model_fsi.losscompute(XbdrIN,label = UbdrIN,LOSS_TYPE = 'Dirichlet')
    LbdrWALL = model_fsi.losscompute(XbdrWALL,label = UbdrWALL,LOSS_TYPE = 'Dirichlet')
    Lobject = model_fsi.losscompute(Xobject,label = Uobject,LOSS_TYPE = 'Dirichlet')
    LbdrOUT = model_fsi.losscompute(XbdrOUT,normal = NbdrOUT,LOSS_TYPE = 'Neutral')

    Opt_u.zero_grad()
    Opt_p.zero_grad()
    loss_list = [Ldomain[0:2],Ldomain[2:3],LbdrIN,LbdrWALL,Lobject,LbdrOUT]
    Losses = []
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
        print("EPOCH : {:5d} \t Total: {:5.8f} \t PDE: {:5.8f} \t DIV: {:5.8f} \t Inlet: {:5.8f} \t BdrDiri: {:5.8f} \t Outlet: {:5.8f}".format(
            epoch_i,Loss_total, Losses[0], Losses[1], Losses[2], Losses[3], Losses[4]))

        if Loss_total<1 and a_ns<1e-3:
            this_N = 0
            a_ns = a_ns*10
            alphas[0] = a_ns
            print("Alpha updated : %2.2e"%a_ns)

        if Loss_total<1 and a_ns>=1e-3 and time_bound<time[-1]:
            this_N = 0
            time_bound = time_bound + time_step
            sampler.update_time(time_bound)

    if epoch_i%2000==0:
        sampler.visualize(N_u,N_p)
        save_cpt(folder_ckp+'{:d}'.format(epoch_i),N_u,N_p)

