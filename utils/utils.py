import os
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
