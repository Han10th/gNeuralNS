import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable

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

class SAMPLER_RECTANGLE(nn.Module):
    def __init__(self, domain, time, visgrid=[20,10,5],device='cpu'):
        super(SAMPLER_RECTANGLE, self).__init__()
        self.Nl, self.Nr, self.Nt = visgrid[0],visgrid[1],visgrid[2]
        self.device = device
        self.time = time        # [Start, VisEnd, End]
        self.domain = domain    # [W, H]

        vis_domain,vis_time = sample_rectangle_grid(visgrid, domain, time)
        self.vis_domain = self.data_warper(vis_domain)
        self.vis_time = self.data_warper(vis_time)

    def forward(self, N):
        # Generate points in fluid domain
        w,h = self.domain[0],self.domain[1]
        T0,T2 = self.time[0],self.time[2]

        domain_t = np.random.uniform(low=T0,    high=T2,    size=(N, 1))
        domain_x = np.random.uniform(low=0,     high=w,     size=(N, 1))
        domain_y = np.random.uniform(low=-h/2,  high=h/2,   size=(N, 1))

        boundary_t=np.random.uniform(low=T0,    high=T2,    size=(N, 1))

        boundL_x =np.random.uniform(low=0,     high=0,     size=(N, 1))
        boundL_y =np.random.uniform(low=-h/2,  high=h/2,   size=(N, 1))

        boundR_x=np.random.uniform(low=w,     high=w,     size=(N, 1))
        boundR_y=np.random.uniform(low=-h/2,  high=h/2,   size=(N, 1))

        boundU_x =np.random.uniform(low=0,     high=w,     size=(N, 1))
        boundU_y =np.random.uniform(low=h/2,   high=h/2,   size=(N, 1))

        boundD_x =np.random.uniform(low=0,     high=w,     size=(N, 1))
        boundD_y =np.random.uniform(low=-h/2,  high=-h/2,  size=(N, 1))

        parabolic_profile = 1 - (boundL_y / (h/2))**2
        boundL_u = 10 * parabolic_profile
        boundL_0 = 0*boundL_y

        Xdomain = np.concatenate((domain_x,domain_y,domain_t),  axis=1)
        XboundL = np.concatenate((boundL_x,boundL_y,boundary_t),axis=1)
        XboundR = np.concatenate((boundR_x,boundR_y,boundary_t),axis=1)
        XboundU = np.concatenate((boundU_x,boundU_y,boundary_t),axis=1)
        XboundD = np.concatenate((boundD_x,boundD_y,boundary_t),axis=1)
        VboundL = np.concatenate((boundL_u,boundL_0), axis=1)
        Vbound0 = np.concatenate((boundL_0,boundL_0), axis=1)

        NboundR = np.concatenate((
            np.ones((N, 1)), np.zeros((N, 1))
        ), axis=1)

        return self.data_warper(Xdomain), self.data_warper(NboundR),\
               self.data_warper(XboundL), self.data_warper(XboundR),\
               self.data_warper(XboundU), self.data_warper(XboundD),\
               self.data_warper(VboundL), self.data_warper(Vbound0),\


    def data_warper(self, data):
        return Variable(torch.from_numpy(data).float(), requires_grad=True).to(self.device)

    def visualize(self,Nu=None,Np=None,Nd=None):
        Nl,Nr,Nt = self.Nl, self.Nr, self.Nt
        N_vec = 10

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
            self.fig = plt.figure(figsize=(8, 10), dpi=150)
        for i in range(self.Nt):
            if self.Nt < 20:
                plt.subplot(int(self.Nt/2), 2, i + 1)
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
                X_current_np[i, ::N_vec, ::N_vec, 0],
                X_current_np[i, ::N_vec, ::N_vec, 1],
                U_current_np[i, ::N_vec, ::N_vec, 0],
                U_current_np[i, ::N_vec, ::N_vec, 1], scale=5
            )

            plt.plot(X_current_np[i, :, 0, 0], X_current_np[i, :, 0, 1], c='r')
            plt.plot(X_current_np[i, :, -1, 0], X_current_np[i, :, -1, 1], c='r')
            plt.scatter(X_current_np[i, :, 0, 0], X_current_np[i, :, 0, 1], s=1, c='b')
            plt.scatter(X_current_np[i, :, -1, 0], X_current_np[i, :, -1, 1], s=1, c='b')

            if self.Nt >= 20:
                plt.axis('off')
                self.fig.savefig('frame/f_{}_{}.png'.format(0, i))
                plt.close(self.fig)
        if self.Nt < 20:
            plt.show()
        #Draw pressure color - x
        #Draw velocity vector