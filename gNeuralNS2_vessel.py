import numpy as np
import torch
from utils.utils import check_folder,save_cpt,load_cpt
from utils.networks import FNN,SFNN
from utils.loss_fsi2 import MODEL_FSI
from utils.sampler_vessel2 import SAMPLER2D_VESSEL
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# File io
folder_ckp = 'node/ns2d_vessel/'
check_folder(folder_ckp)
# Visualization Parameter
visgrid = [500,51,4]
# Region definition
vessel_connect = np.array([
])
vessel_pts = np.array([
    [0, 0], [0, 4], [4, 4]#, [6, 15], [10, 14], [12, 10], [12, 5]
])
vessel_radius = np.array([
    1,1,1,1,1,1,1
])
vessel_smoother = np.array([
    0,1,0#1,1,1,1,0
])*2
time = np.array([0,0,1,1])
# time = [0,0,2,2]
# Seq2Seq parameter
sequence_N = 50
time_step = (time[-1] + time[0]) / sequence_N
time_bound = time[0] + time_step
# Solid Parameter
rho_s = 1.3
h_0 = 0.1
E = 1e5
xi = 0.5
R0 = 1
cof_a = E / (rho_s * (1-xi*xi) * R0*R0)
cof_H = rho_s * h_0
# Fluid Parameter
Re = 100
vessel_RRC = [
    [],
    [0.118, 1.117, 1.016],
]
# Networks Parameter
epoch_N = 100000
sample_N = 1000
lr_u = 1e-4
lr_p = 1e-3
lr_s = 1e-3
a_ns = 1e-4
alphas = [a_ns, 10, 1, 10, 1]
# Alpha_ns dynamic weighting
N_loss = len(alphas)
this_N = 0

N_u = FNN([3] + 10 * [40] + [2], "tanh", "Glorot normal").to(device)
N_p = FNN([3] + 10 * [20] + [1], "tanh", "Glorot normal").to(device)
N_s = FNN([3] + 10 * [20] + [2], "tanh", "zero").to(device)
load_cpt(folder_ckp+'{:d}'.format(1),N_u,N_p)
Opt_u = torch.optim.Adam(list(N_u.parameters()), lr=lr_u)
Opt_p = torch.optim.Adam(list(N_p.parameters()), lr=lr_p)
Opt_s = torch.optim.Adam(list(N_s.parameters()), lr=lr_s)
model_fsi = MODEL_FSI(N_u, N_p, N_s, Re, cof_a, cof_H)

sampler = SAMPLER2D_VESSEL(vessel_pts, vessel_radius, vessel_smoother,time, visgrid, device)
# sampler.update_time(time_bound)
# sampler.visualize_sampled_domain()

for epoch_i in range(epoch_N):
    this_N = this_N + 1
    Xdomain = sampler.sample_domain(sample_N)
    XbdrIN,     UbdrIN,     NbdrIN  = sampler.sample_inlet(sample_N)
    XbdrWALL,   UbdrWALL,   NbdrWALL= sampler.sample_wall(sample_N)
    XbdrOUT,    UbdrOUT,    NbdrOUT = sampler.sample_outlet(sample_N)

    Ldomain = model_fsi.losscompute(Xdomain,LOSS_TYPE = 'PDE')
    LbdrIN = model_fsi.losscompute(XbdrIN,label = UbdrIN,LOSS_TYPE = 'Dirichlet')
    ### NEED TO BE COMPLETED ###
    LbdrWALL = model_fsi.losscompute(XbdrWALL,label = UbdrWALL,LOSS_TYPE = 'Dirichlet')
    LbdrOUT = model_fsi.losscompute(XbdrOUT,normal = NbdrOUT,LOSS_TYPE = 'Neutral')

    Opt_u.zero_grad()
    Opt_p.zero_grad()
    loss_list = [Ldomain[0:2], Ldomain[2:3], LbdrIN, LbdrWALL, LbdrOUT]
    Losses = []
    Loss_total = 0
    for i in range(N_loss):
        Loss = 0
        for loss in loss_list[i]:
            Loss += torch.mean(torch.square(torch.square(loss)))
        Losses += [Loss]
        Loss_total += alphas[i] * Losses[i]
    Loss_total.backward()
    Opt_u.step()
    Opt_p.step()

    if epoch_i % 100 == 0:
        print(
            "EPOCH : {:5d} \t Total: {:5.8f} \t PDE: {:5.8f} \t DIV: {:5.8f} \t Inlet: {:5.8f} \t BdrDiri: {:5.8f} \t Outlet: {:5.8f}".format(
                epoch_i, Loss_total, Losses[0], Losses[1], Losses[2], Losses[3], Losses[4]))

        if Loss_total < 1 and a_ns < 1e-3:
            this_N = 0
            a_ns = a_ns * 10
            alphas[0] = a_ns
            print("Alpha updated : %2.2e" % a_ns)

        if Loss_total < 1 and a_ns >= 1e-3 and time_bound < time[-1]:
            this_N = 0
            time_bound = time_bound + time_step
            # sampler.update_time(time_bound)

    if epoch_i % 2000 == 0:
        sampler.visualize(N_u, N_p, N_s)
        # sampler.visualize(N_u, N_p, N_s,model_fsi.losscompute)
        save_cpt(folder_ckp + '{:d}'.format(epoch_i), N_u, N_p, N_s)