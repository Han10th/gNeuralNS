import os
import numpy as np
import torch

def differential_y_x(y,X,component):
    return torch.autograd.grad(y.sum(), X, create_graph=True)[0][:,component:component+1]

def check_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_cpt(file_suffix,N_u,N_p,N_s=None):
    torch.save(N_u.state_dict(), file_suffix + 'NuEp')
    torch.save(N_p.state_dict(), file_suffix + 'NpEp')
    if N_s is not None:
        torch.save(N_s.state_dict(), file_suffix + 'NsEp')
    print(file_suffix + 'checkpoints : SAVE COMPLETED!')

def load_cpt(file_suffix,N_u,N_p,N_s=None):
    N_u.load_state_dict(torch.load(file_suffix + 'NuEp'))
    N_p.load_state_dict(torch.load(file_suffix + 'NpEp'))
    if N_s is not None:
        N_s.load_state_dict(torch.load(file_suffix + 'NsEp'))
    print(file_suffix + 'checkpoints : LOAD COMPLETED!')
def load_cpt1d(file_suffix,N_Q_list, N_P_list):
    N_Q_list = (torch.load(file_suffix + 'NQlistEp'))
    N_P_list = (torch.load(file_suffix + 'NPlistEp'))
    print(file_suffix + 'checkpoints : LOAD COMPLETED!')
    return N_Q_list,N_P_list
def save_cpt1d(file_suffix,N_Q_list, N_P_list):
    torch.save(N_Q_list, file_suffix + 'NQlistEp')
    torch.save(N_P_list, file_suffix + 'NPlistEp')
    print(file_suffix + 'checkpoints : SAVE COMPLETED!')

def ExtractParameters(N_list):
    Parameters=list()
    for N in N_list:
        Parameters += list(N.parameters())
    return Parameters
def MakeOptimizers(N_list,lr):
    Opt_list = []
    for N in N_list:
        Opt_list += [torch.optim.Adam(list(N.parameters()), lr=lr)]
    return Opt_list
def ClearGradients(Opt_list):
    for Opt in Opt_list:
        Opt.zero_grad()
def StepGradients(Opt_list):
    for Opt in Opt_list:
        Opt.step()
def ToTensor(data,device):
    return torch.autograd.Variable(torch.from_numpy(data).float(), requires_grad=True).to(device)
def is_leaf(Connect,j):
    return np.sum(Connect[:,j])==0

def pair_domainVStime(domain, time, Nl,Nr,Nt):
    domain = np.reshape(
        np.tile(domain, (Nt, 1, 1, 1)), (Nt * Nl * Nr, 2)
    )
    time = np.reshape(
        np.tile(time, (1, Nl * Nr)), (Nt * Nl * Nr, 1)
    )
    return domain, time