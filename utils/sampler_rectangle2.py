import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable

Velocity = 20
def sample_rectangle_grid(grid,domain,time):
    Nl, Nr, Nt = grid[0], grid[1], grid[2]
    w,h = domain[0], domain[1]
    x = np.tile(np.expand_dims(np.linspace(0,   w,  Nl), axis=(0,2)),(Nr,1,1))
    y = np.tile(np.expand_dims(np.linspace(-h/2,h/2,Nr), axis=(1,2)),(1,Nl,1))

    vis_domain = np.concatenate((x,y), axis=-1)
    vis_time = np.expand_dims(np.linspace(time[0], time[1], Nt), axis=1)

    vis_domain = np.reshape(
        np.tile(vis_domain, (Nt, 1, 1, 1)), (Nt * Nl * Nr, 2)
    )
    vis_time = np.reshape(
        np.tile(vis_time, (1, Nl * Nr)), (Nt * Nl * Nr, 1)
    )
    return vis_domain,vis_time

def generate_parabolic_v(y,t,h):
    parabolic_profile = 1 - (y / (h / 2)) ** 2
    # v = (1 - np.cos(2*np.pi*t)) / 2
    v = 1
    bdrL_u = Velocity * parabolic_profile * v
    bdrL_0 = 0 * y
    return bdrL_u,bdrL_0

class SAMPLER_RECTANGLE(nn.Module):
    def __init__(self, domain, time, visgrid=[20,10,5],device='cpu', func_object=None):
        super(SAMPLER_RECTANGLE, self).__init__()
        self.Nl, self.Nr, self.Nt = visgrid[0],visgrid[1],visgrid[2]
        self.device = device
        self.time = time        # [Start, VisEnd, End]
        self.domain = domain    # [W, H]

        vis_domain,vis_time = sample_rectangle_grid(visgrid, domain, time)
        self.vis_domain = self.data_warper(vis_domain)
        self.vis_time = self.data_warper(vis_time)

        self.func_object = func_object
        if func_object is not None:
            self.vis_obj_bdr = self.func_object.generate_boundary(visgrid)

    def forward(self, N):
        # Generate points in fluid domain
        w,h = self.domain[0],self.domain[1]
        T0,T2 = self.time[0],self.time[2]

        domain_t = np.random.uniform(low=T0,    high=T2,    size=(5*N, 1))
        domain_x = np.random.uniform(low=0,     high=w,     size=(5*N, 1))
        domain_y = np.random.uniform(low=-h/2,  high=h/2,   size=(5*N, 1))

        boundary_t=np.random.uniform(low=T0,    high=T2,    size=(N, 1))

        bdrL_x = np.random.uniform(low=0,     high=0,     size=(N, 1))
        bdrL_y = np.random.uniform(low=-h/2,  high=h/2,   size=(N, 1))

        bdrR_x = np.random.uniform(low=w,     high=w,     size=(N, 1))
        bdrR_y = np.random.uniform(low=-h/2,  high=h/2,   size=(N, 1))

        bdrU_x = np.random.uniform(low=0,     high=w,     size=(N, 1))
        bdrU_y = np.random.uniform(low=h/2,   high=h/2,   size=(N, 1))

        bdrD_x = np.random.uniform(low=0,     high=w,     size=(N, 1))
        bdrD_y = np.random.uniform(low=-h/2,  high=-h/2,  size=(N, 1))

        bdrL_u, bdrL_0 = generate_parabolic_v(bdrL_y, boundary_t, h)
        Xdomain = np.concatenate((domain_x,domain_y,domain_t),  axis=1)
        XbdrDiri1 = ((
            np.concatenate((bdrL_x,bdrL_y,boundary_t),axis=1)
        ))
        UbdrDiri1 = ((
            np.concatenate((bdrL_u,bdrL_0), axis=1)
        ))
        XbdrDiri0 = np.concatenate((
            np.concatenate((bdrU_x,bdrU_y,boundary_t),axis=1),
            np.concatenate((bdrD_x, bdrD_y, boundary_t), axis=1)
        ),axis=0)
        UbdrDiri0 = np.concatenate((
            np.concatenate((bdrL_0,bdrL_0), axis=1),
            np.concatenate((bdrL_0,bdrL_0), axis=1)
        ),axis=0)
        XbdrNeutral = ((
            np.concatenate((bdrR_x,bdrR_y,boundary_t),axis=1)
        ))
        NbdrNeutral = ((
            np.concatenate((np.ones((N, 1)), np.zeros((N, 1))), axis=1)
        ))

        if self.func_object is not None:
            Xdisk = self.func_object.generate_domain(N)
            # Xdisk = Xdomain[self.func_object.inside_ellipse(Xdomain),:]
            Xdomain = Xdomain[~self.func_object.inside_ellipse(Xdomain),:]
            XbdrDiri0 = np.concatenate((
                XbdrDiri0,
                Xdisk
            ),axis=0)
            UbdrDiri0 = np.concatenate((
                UbdrDiri0,
                Xdisk[:,0:2]*0
            ),axis=0)

        return self.data_warper(Xdomain),\
               self.data_warper(XbdrDiri1), self.data_warper(UbdrDiri1),\
               self.data_warper(XbdrDiri0), self.data_warper(UbdrDiri0),\
               self.data_warper(XbdrNeutral), self.data_warper(NbdrNeutral),\


    def data_warper(self, data):
        return Variable(torch.from_numpy(data).float(), requires_grad=True).to(self.device)

    def visualize(self,Nu=None,Np=None,Nd=None):
        Nl,Nr,Nt = self.Nl, self.Nr, self.Nt
        N_step = 4

        X_reference = torch.concat((
            self.vis_domain,self.vis_time
        ),dim=1)

        # D_current = Nd(self.X_reference)
        # Xt = self.vis_domain + D_current
        X_current = X_reference
        U_current = Nu(X_current)
        P_current = Np(X_current)

        X_current_np = np.reshape(X_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl,3))
        U_current_np = np.reshape(U_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl,2))
        P_current_np = np.reshape(P_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl))

        Color = np.sqrt(np.sum(U_current_np ** 2, axis=-1))
        ColorMax, ColorMin = Color.max(), Color.min()

        if self.Nt < 20:
            self.fig = plt.figure(figsize=(5, 10), dpi=150)
        for i in range(self.Nt):
            if self.Nt < 20:
                plt.subplot(self.Nt, 1, i + 1)
            else:
                self.fig = plt.figure(figsize=(8, 10), dpi=150)

            plt.gca().axis('equal')
            plt.gca().pcolormesh(
                X_current_np[i, :, :, 0],
                X_current_np[i, :, :, 1],
                Color[i, :, :],
                vmin=ColorMin, vmax=ColorMax
            )
            plt.quiver(
                X_current_np[i, ::N_step, ::N_step, 0],
                X_current_np[i, ::N_step, ::N_step, 1],
                U_current_np[i, ::N_step, ::N_step, 0],
                U_current_np[i, ::N_step, ::N_step, 1], scale=Velocity
            )
            plt.gca().set_xlim([-0.1*self.domain[0],1.1*self.domain[0]])
            plt.gca().set_ylim([-0.6*self.domain[1],0.6*self.domain[1]])
            # # Plot for vessel boundary
            # plt.plot(X_current_np[i, :, 0, 0], X_current_np[i, :, 0, 1], c='r')
            # plt.plot(X_current_np[i, :, -1, 0], X_current_np[i, :, -1, 1], c='r')
            # plt.scatter(X_current_np[i, :, 0, 0], X_current_np[i, :, 0, 1], s=1, c='b')
            # plt.scatter(X_current_np[i, :, -1, 0], X_current_np[i, :, -1, 1], s=1, c='b')

            if self.func_object is not None:
                plt.plot(self.vis_obj_bdr[i, :, 0], self.vis_obj_bdr[i, :, 1], 'r')

            if self.Nt >= 20:
                plt.axis('off')
                self.fig.savefig('frame/f_{}_{}.png'.format(0, i))
                plt.close(self.fig)
        if self.Nt < 20:
            plt.show()
        #Draw pressure color - x
        #Draw velocity vector