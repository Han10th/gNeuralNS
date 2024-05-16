import numpy as np
import torch
from utils.utils import check_folder,save_cpt,load_cpt
from utils.networks import FNN,SFNN
from utils.domain2 import ELLIPSE
from utils.loss_fsi2 import MODEL_FSI
from utils.sampler_rectangle2 import SAMPLER_RECTANGLE
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# File io
folder_ckp = 'node/ns1d_single/'
check_folder(folder_ckp)
# Visualization Parameter
visgrid = [40, 20, 5]
# Region definition
key_pts = [0,2,4]
key_radius = [1,1,1]
time = [0, 1]
# Solid Parameter
# Fluid Parameter
# Networks Parameter
epoch_N = 100000
sample_N = 1000
lr_Q = 1e-3
lr_P = 1e-3
lr_A = 1e-3
a_ns = 1e-7
alphas = [a_ns, 10, 1, 1, 1]
# Alpha_ns dynamic weighting
N_loss = len(alphas)
this_N = 0

N_Q = FNN([3] + 10 * [20] + [1], "tanh", "Glorot normal").to(device)
N_P = FNN([3] + 10 * [20] + [1], "tanh", "Glorot normal").to(device)
N_A = FNN([3] + 10 * [20] + [1], "tanh", "zero").to(device)
# load_cpt(folder_ckp+'{:d}'.format(-10),N_u,N_p)
Opt_Q = torch.optim.Adam(list(N_Q.parameters()), lr=lr_Q)
Opt_P = torch.optim.Adam(list(N_P.parameters()), lr=lr_P)
Opt_A = torch.optim.Adam(list(N_A.parameters()), lr=lr_A)


model_fsi = MODEL_FSI(N_u, N_p)