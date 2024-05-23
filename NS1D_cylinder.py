import numpy as np
import torch
from utils.utils import check_folder,save_cpt,load_cpt,ExtractParameters
from utils.networks import multiple_FNN
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
    # [0]
    [0,0,0],
    [1,0,0],
    [1,0,0]
])
vessel_pts = np.array([
    [0,0,0],
    [20,0,0],
    [40,10,0],
    [40,-10,0]
])
vessel_radius = np.array([
    [2,1],
    [1,1],
    [1,1]
])
N_segment = len(vessel_radius)
time = np.array([0,0,1,1])
# Solid Parameter
E = 1e6
h0 = 0.1
beta = E*h0*np.sqrt(np.pi)
# Fluid Parameter
vessel_RRC = [
    [],
    [0.118, 1.117, 1.016],
    [0.118, 1.117, 1.016]
]
alpha=1.3
rho_f = 1.06
nu=0.035
Kr = 8*np.pi*nu
# Networks Parameter
epoch_N = 100000
sample_N = 1000
lr_Q = 1e-3
lr_P = 1e-3
lr_R = 1e-3
a_ns = 1e-2
alphas = [a_ns, 1, 1, 10, 1, 1]
# Alpha_ns dynamic weighting
N_loss = len(alphas)
this_N = 0

sampler = SAMPLER_VESSEL(vessel_connect, vessel_pts, vessel_radius,
                         time, visgrid, device=device)
N_Q_list = multiple_FNN(N_segment,[2] + 5 * [20] + [1], "tanh", "Glorot normal",device)
N_P_list = multiple_FNN(N_segment,[2] + 5 * [20] + [1], "tanh", "Glorot normal",device)
N_R_list = multiple_FNN(N_segment,[2] + 5 * [20] + [1], "tanh", "zero",device)
# load_cpt(folder_ckp+'{:d}'.format(-10),N_u,N_p)
Opt_Q = torch.optim.Adam(ExtractParameters(N_Q_list), lr=lr_Q)
Opt_P = torch.optim.Adam(ExtractParameters(N_P_list), lr=lr_P)
Opt_R = torch.optim.Adam(ExtractParameters(N_R_list), lr=lr_R)
model_ns1d = MODEL_NS1D(sampler, N_Q_list, N_P_list, N_R_list,alpha,beta,rho_f,Kr,RRC=vessel_RRC)

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

        if Loss_total < 2 and a_ns < 1:
            this_N = 0
            a_ns = a_ns * 10
            alphas[0] = a_ns
            print("Alpha updated : %2.2e" % a_ns)

    if epoch_i % 2000 == 0:
        sampler.visualize(N_Q_list, N_P_list, N_R_list)
        # save_cpt(folder_ckp + '{:d}'.format(epoch_i), N_u, N_p, N_s)
