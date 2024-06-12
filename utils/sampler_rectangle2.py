import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import ToTensor,pair_domainVStime

Velocity = 10
def generate_rectangle_grid(grid,domain,time):
    Nl, Nr, Nt = grid[0], grid[1], grid[2]
    w,h = domain[0], domain[1]
    x = np.tile(np.expand_dims(np.linspace(0,   w,  Nl), axis=(0,2)),(Nr,1,1))
    y = np.tile(np.expand_dims(np.linspace(-h/2,h/2,Nr), axis=(1,2)),(1,Nl,1))

    vis_domain = np.concatenate((x,y), axis=-1)
    vis_time = np.expand_dims(np.linspace(time[1], time[2], Nt), axis=1)

    return pair_domainVStime(vis_domain,vis_time)

def generate_parabolic_v(y,t,h):
    parabolic_profile = 1 - (y / (h / 2)) ** 2
    # v = (1 - np.cos(2*np.pi*t)) / 2
    v = 1
    bdrL_u = Velocity * parabolic_profile * v
    bdrL_0 = 0 * y
    return bdrL_u,bdrL_0

class SAMPLER_RECTANGLE:
    def __init__(self, domain, time=[0,0,0.5,1], visgrid=[20,10,5],device='cpu', func_objects=[]):
        super(SAMPLER_RECTANGLE, self).__init__()
        self.Nl, self.Nr, self.Nt = visgrid[0],visgrid[1],visgrid[2]
        self.device = device
        self.time = np.array(time)        # [Start, VisStart, VisEnd, End]
        self.domain = np.array(domain)    # [W, H]

        vis_domain,vis_time = generate_rectangle_grid(visgrid, domain, time)
        self.vis_domain = self.data_warper(vis_domain)
        self.vis_time = self.data_warper(vis_time)

        self.func_objects = func_objects
        if self.func_objects:
            self.vis_obj_bdr = []
            self.vel_obj_bdr = []
            for func_object in self.func_objects:
                X_vis_obj,U_vis_obj = func_object.generate_boundary(visgrid)
                self.vis_obj_bdr += [X_vis_obj]
                self.vel_obj_bdr += [U_vis_obj]

    # def forward(self):
    #     print("NOTHING")
    def sample_domain(self,N, t=None):
        w,h = self.domain[0],self.domain[1]
        T0,T3 = self.time[0],self.time[3]

        domain_t = np.random.uniform(low=T0, high=T3, size=(5 * N, 1))
        if t is not None:
            domain_t = domain_t * 0 + t
        domain_x = np.random.uniform(low=0,     high=w,     size=(5*N, 1))
        domain_y = np.random.uniform(low=-h/2,  high=h/2,   size=(5*N, 1))
        Xdomain = np.concatenate((domain_x,domain_y,domain_t),  axis=1)
        if self.func_objects:
            for func_object in self.func_objects:
                Xdomain = Xdomain[~func_object.inside_ellipse(Xdomain),:]
        return self.data_warper(Xdomain)
    def sample_inlet(self,N):
        w,h = self.domain[0],self.domain[1]
        T0,T3 = self.time[0],self.time[3]

        boundary_t=np.random.uniform(low=T0,    high=T3,    size=(N, 1))
        bdrL_x = np.random.uniform(low=0,     high=0,     size=(N, 1))
        bdrL_y = np.random.uniform(low=-h/2,  high=h/2,   size=(N, 1))
        bdrL_u, bdrV_0 = generate_parabolic_v(bdrL_y, boundary_t, h)
        XbdrIN = np.concatenate((bdrL_x,bdrL_y,boundary_t),axis=1)
        UbdrIN = np.concatenate((bdrL_u,bdrV_0), axis=1)
        NbdrIN = np.concatenate((-np.ones((N, 1)), np.zeros((N, 1))), axis=1)
        return self.data_warper(XbdrIN), self.data_warper(UbdrIN), self.data_warper(NbdrIN)
    def sample_wall(self,N):
        w,h = self.domain[0],self.domain[1]
        T0,T3 = self.time[0],self.time[3]

        boundary_t=np.random.uniform(low=T0,    high=T3,    size=(N, 1))
        bdrU_x = np.random.uniform(low=0,     high=w,     size=(N, 1))
        bdrU_y = np.random.uniform(low=h/2,   high=h/2,   size=(N, 1))
        bdrD_x = np.random.uniform(low=0,     high=w,     size=(N, 1))
        bdrD_y = np.random.uniform(low=-h/2,  high=-h/2,  size=(N, 1))
        XbdrWALL = np.concatenate((
            np.concatenate((bdrU_x,bdrU_y,boundary_t),axis=1),
            np.concatenate((bdrD_x, bdrD_y, boundary_t), axis=1)
        ),axis=0)
        UbdrWALL = np.concatenate((
            np.concatenate((np.zeros((N, 1)),np.zeros((N, 1))), axis=1),
            np.concatenate((np.zeros((N, 1)),np.zeros((N, 1))), axis=1)
        ),axis=0)
        NbdrWALL = np.concatenate((
            np.concatenate((np.zeros((N, 1)), np.ones((N, 1))), axis=1),
            np.concatenate((np.zeros((N, 1)),-np.ones((N, 1))), axis=1)
        ),axis=0)
        return self.data_warper(XbdrWALL), self.data_warper(UbdrWALL), self.data_warper(NbdrWALL)
    def sample_outlet(self,N):
        w,h = self.domain[0],self.domain[1]
        T0,T3 = self.time[0],self.time[3]

        bdrR_x = np.random.uniform(low=w,     high=w,     size=(N, 1))
        bdrR_y = np.random.uniform(low=-h/2,  high=h/2,   size=(N, 1))
        boundary_t=np.random.uniform(low=T0,    high=T3,    size=(N, 1))
        XbdrOUT = np.concatenate((bdrR_x,bdrR_y,boundary_t),axis=1)
        UbdrOUT = np.concatenate((np.zeros((N, 1)),np.zeros((N, 1))), axis=1)
        NbdrOUT = np.concatenate(( np.ones((N, 1)),np.zeros((N, 1))), axis=1)
        return self.data_warper(XbdrOUT), self.data_warper(UbdrOUT), self.data_warper(NbdrOUT)
    def sample_objects(self,N, t=None):
        X_disk = np.zeros((0,3))
        V_disk = np.zeros((0,2))
        for func_object in self.func_objects:
            X_domain = func_object.sample_domain(N, t)
            V_domain = X_domain[:, 0:2] * 0
            X_boundary,V_boundary = func_object.sample_boundary(N, t)
            X_disk = np.concatenate((
                X_disk,X_boundary#,X_domain
            ),axis=0)
            V_disk = np.concatenate((
                V_disk,V_boundary#,V_domain
            ),axis=0)
        return self.data_warper(X_disk), self.data_warper(V_disk)

    def data_warper(self, data):
        return ToTensor(data,self.device)
    def visualize_sampled_domain(self, domain):
        T1,T2 = self.time[1],self.time[2]
        N = 1000
        Nt = 5
        N_step = 10
        tstep=(T2-T1)/(Nt-1)
        plt.figure(figsize=(5, 10), dpi=150)
        for i in range(Nt):
            time = tstep*i
            plt.subplot(5, 1, i+1)
            plt.gca().axis('equal')
            plt.gca().set_xlim([-0.1*domain[0],1.1*domain[0]])
            plt.gca().set_ylim([-0.6*domain[1],0.6*domain[1]])
            dom = self.sample_domain(N, time)
            dom = dom.cpu().detach().numpy()
            plt.scatter(dom[:, 0], dom[:, 1], s=1, c='b')
            for (vis_obj_bdr,vel_obj_bdr) in zip(self.vis_obj_bdr,self.vel_obj_bdr):
                plt.plot(vis_obj_bdr[i, :, 0], vis_obj_bdr[i, :, 1], 'r')
                plt.quiver(
                    vis_obj_bdr[i, ::N_step, 0],
                    vis_obj_bdr[i, ::N_step, 1],
                    vel_obj_bdr[i, ::N_step, 0],
                    vel_obj_bdr[i, ::N_step, 1], scale=Velocity
                )


            if self.func_objects:
                obj,vel = self.sample_objects(N, time)
                obj = obj.cpu().detach().numpy()
                vel = vel.cpu().detach().numpy()
                plt.scatter(obj[:, 0], obj[:, 1], s=1, c='r')
                plt.quiver(
                    obj[::N_step, 0],
                    obj[::N_step, 1],
                    vel[::N_step, 0],
                    vel[::N_step, 1], scale=Velocity
                )
        plt.show()
    def visualize(self,Nu=None,Np=None,Ns=None):
        Nl,Nr,Nt = self.Nl, self.Nr, self.Nt
        N_step = 4
        N_stream = 10
        Hy = self.domain[1]/2 * 0.8
        stream_points = np.concatenate((
            np.zeros((N_stream,1)),
            np.expand_dims(np.linspace(Hy,-Hy,N_stream),axis=1)
        ),axis=1)

        X_reference = torch.concat((
            self.vis_domain,self.vis_time
        ),dim=1)

        if Ns is not None:
            D_current = Ns(X_reference)
            X_current = torch.concat((
                X_reference[:,0:1],
                X_reference[:,1:2]+D_current,
                X_reference[:,-1:]
            ),axis=1)
        else:
            X_current = X_reference
        U_current = Nu(X_current)
        P_current = Np(X_current)

        X_current_np = np.reshape(X_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl,3))
        U_current_np = np.reshape(U_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl,2))
        P_current_np = np.reshape(P_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl))

        Color = np.sqrt(np.sum(U_current_np ** 2, axis=-1))
        ColorMax, ColorMin = Color.max(), Color.min()

        if self.Nt < 20:
            self.fig = plt.figure(figsize=(4, 10), dpi=150)
        for i in range(self.Nt):
            if self.Nt < 20:
                plt.subplot(self.Nt, 1, i + 1)
            else:
                self.fig = plt.figure(figsize=(8, 4), dpi=150)
            plt.tight_layout()
            plt.gca().axis('equal')
            plt.gca().pcolormesh(
                X_current_np[i, :, :, 0],
                X_current_np[i, :, :, 1],
                Color[i, :, :],
                vmin=ColorMin, vmax=ColorMax
            )
            # plt.quiver(
            #     X_current_np[i, ::N_step, ::N_step, 0],
            #     X_current_np[i, ::N_step, ::N_step, 1],
            #     U_current_np[i, ::N_step, ::N_step, 0]/Velocity,
            #     U_current_np[i, ::N_step, ::N_step, 1]/Velocity, scale=Velocity
            # )
            plt.streamplot(
                X_current_np[i, ::N_step, ::N_step, 0],
                X_current_np[i, ::N_step, ::N_step, 1],
                U_current_np[i, ::N_step, ::N_step, 0],
                U_current_np[i, ::N_step, ::N_step, 1],
                # start_points=stream_points, density=[0.5, 1])
                density=0.6, color='k', linewidth=Color[i, ::N_step, ::N_step]/Velocity)

            # # Plot for vessel boundary
            # plt.plot(X_current_np[i, :, 0, 0], X_current_np[i, :, 0, 1], c='r')
            # plt.plot(X_current_np[i, :, -1, 0], X_current_np[i, :, -1, 1], c='r')
            # plt.scatter(X_current_np[i, :, 0, 0], X_current_np[i, :, 0, 1], s=1, c='b')
            # plt.scatter(X_current_np[i, :, -1, 0], X_current_np[i, :, -1, 1], s=1, c='b')

            if self.func_objects:
                for vis_obj_bdr in self.vis_obj_bdr:
                    Xobj_current = self.data_warper(vis_obj_bdr[i, :, :])
                    Uobj_current = Nu(Xobj_current)
                    Uobj_current = Uobj_current.detach().cpu().numpy()
                    plt.plot(vis_obj_bdr[i, :, 0], vis_obj_bdr[i, :, 1], 'r')
                    # plt.quiver(
                    #     vis_obj_bdr[i, ::N_step, 0],
                    #     vis_obj_bdr[i, ::N_step, 1],
                    #     Uobj_current[::N_step, 0]/Velocity,
                    #     Uobj_current[::N_step, 1]/Velocity, scale=Velocity
                    # )

            plt.gca().set_xlim([-0.1,0.1+self.domain[0]])
            plt.gca().set_ylim([-0.6*self.domain[1],0.6*self.domain[1]])
            if self.Nt >= 20:
                plt.axis('off')
                self.fig.savefig('frame/f_{}_{}.png'.format(0, i))
                plt.close(self.fig)
        if self.Nt < 20:
            plt.show()
        #Draw pressure color - x
        #Draw velocity vector
    def update_time(self, time_bound):
        self.time[-1] = time_bound
        print("[S2S] SAMPLER time limit updated to {:2.2f}".format(time_bound))