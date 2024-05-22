import numpy as np
import torch
from utils.utils import check_folder,save_cpt,load_cpt
from utils.networks import FNN,SFNN
from utils.loss_NS1 import MODEL_NS1D
from utils.sampler_vessel1 import SAMPLER_VESSEL,compute_preprocess
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
    The vessel tree of N segments is recorded by 
    vessel_connect (connectivity, NxN) and vessel_pts (key points, (N+1)x3).
    The first segment is (vessel_pts[0,:],vessel_pts[1,:])
    The j-th segments are (vessel_pts[i+1,:],vessel_pts[j+1,:]) where vessel_connect[i,j]==1
'''
# File io
folder_ckp = 'node/ns1d_single/'
check_folder(folder_ckp)
# Visualization Parameter
visgrid = [40, 5]
# Region definition
vessel_connect = np.array([
    [0,0],
    [1,0]
])
vessel_pts = np.array([
    [0,0,0],
    [20,0,0],
    [40,20,0]
])
vessel_radius = np.array([
    [2,1],
    [1,2]
])
time = np.array([0,0,1,1])
# Solid Parameter
# Fluid Parameter
vessel_RRC = [
    [],
    [0.118, 1.117, 1.016],
]
# Networks Parameter
epoch_N = 100000
sample_N = 1000
lr_Q = 1e-3
lr_P = 1e-3
lr_R = 1e-3
a_ns = 1e-7
alphas = [1, 1, 1, 1, 1, 1]
# Alpha_ns dynamic weighting
N_loss = len(alphas)
this_N = 0

sampler = SAMPLER_VESSEL(vessel_connect, vessel_pts, vessel_radius,
                         time, visgrid, device=device)
N_Q = FNN([2] + 10 * [20] + [1], "tanh", "Glorot normal").to(device)
N_P = FNN([2] + 10 * [20] + [1], "tanh", "Glorot normal").to(device)
N_R = FNN([2] + 10 * [20] + [1], "tanh", "zero").to(device)
# load_cpt(folder_ckp+'{:d}'.format(-10),N_u,N_p)
Opt_Q = torch.optim.Adam(list(N_Q.parameters()), lr=lr_Q)
Opt_P = torch.optim.Adam(list(N_P.parameters()), lr=lr_P)
Opt_R = torch.optim.Adam(list(N_R.parameters()), lr=lr_R)
model_ns1d = MODEL_NS1D(sampler, N_Q, N_P, N_R,RRC=vessel_RRC)

for epoch_i in range(epoch_N):
    this_N = this_N + 1

    Losses = model_ns1d.losscompute_all(sample_N)
    Loss_total = 0
    Opt_Q.zero_grad()
    Opt_P.zero_grad()
    Opt_R.zero_grad()

    for i in range(N_loss):
        Loss_total += alphas[i] * Losses[i]
    Loss_total.backward()
    Opt_Q.step()
    Opt_P.step()
    # Opt_R.step()

    if epoch_i % 100 == 0:
        print(
            "EPOCH : {:5d} \t Total: {:5.8f} \t PDE1: {:5.8f} \t PDE2: {:5.8f} \t Inlet: {:5.8f} \t BranchQ: {:5.8f} \t BranchP: {:5.8f} \t Outlet: {:5.8f}".format(
                epoch_i, Loss_total, Losses[0], Losses[1], Losses[2], Losses[3], Losses[4], Losses[5]))

    if epoch_i % 2000 == 0:
        sampler.visualize(N_Q, N_P, N_R)
        # save_cpt(folder_ckp + '{:d}'.format(epoch_i), N_u, N_p, N_s)
