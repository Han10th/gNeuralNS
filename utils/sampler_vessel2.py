import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import ToTensor,pair_domainVStime
Velocity = 10
def normalize_byrow(vectors):
    # NEED TO VERIFY IF IT'S ROW NORMALIZATION
    return vectors / np.sqrt(np.sum(vectors ** 2, axis=1, keepdims=True))
def compute_angle(vector1, vector2):
    alpha0 = np.arctan2(vector1[1], vector1[0])
    alpha1 = np.arctan2(vector2[1], vector2[0])
    theta = np.mod(alpha1 - alpha0, 2 * np.pi)
    theta_full = theta-2*np.pi if theta>np.pi else theta
    alpha_mid = alpha0 + theta_full/2
    return theta_full, alpha_mid
def compute_normal_pts(p0,p1):
    # Need to check whether this method can always preserve the relative orientation
    v01 = p1-p0
    v01 = v01 / np.sqrt(np.sum(v01**2))
    normal = np.array([-v01[1],v01[0]])
    return normal
def compute_normal_line(line):
    n = line.shape[0]
    normal = np.zeros((n,2,2))
    line_tmp = np.concatenate((2*line[0:1,:]-line[1:2,:],
                               line,
                               2*line[-1:,:]-line[-2:-1,:]),axis=0)
    for i in range(n):
        normal[i:i+1,0,:] = compute_normal_pts(line_tmp[i],line_tmp[i+1])
        normal[i:i+1,1,:] = compute_normal_pts(line_tmp[i+1],line_tmp[i+2])
    return normal
def compute_smoothCL(line,radius,smoother_CL,N=50):
    n = line.shape[0]
    s = np.expand_dims(np.linspace(0, 1, N), axis=[1])
    normal = compute_normal_line(line)

    line_refined = [line[0:1,:]]
    normal_refined = [normal[0:1,0,:]]
    radius_refined = [radius[0:1]]
    for i in range(1,n-1):
        this_pts = line[i,:]
        this_smoother = smoother_CL[i]
        vec_start, vec_end = normal[i,0,:],normal[i,1,:]

        theta_full, alpha_mid = compute_angle(vec_start, vec_end)
        vec_middle = np.array([np.cos(alpha_mid),np.sin(alpha_mid)])
        pts_center = this_pts - this_smoother/np.cos(np.abs(theta_full)/2) * vec_middle
        vec_refined = ((1-s)*vec_start + s*vec_end)
        vec_refined = normalize_byrow(vec_refined)

        line_refined.append(pts_center + this_smoother*vec_refined)
        normal_refined.append(vec_refined)
        radius_refined.append(np.repeat(radius[i:i+1],N))

    line_refined.append(line[-1:,:])
    normal_refined.append(normal[-1:,0,:])
    radius_refined.append(radius[-1:])

    return (np.concatenate(line_refined,axis=0),
            np.concatenate(normal_refined,axis=0),
            np.concatenate(radius_refined,axis=0))
def parameterize_CL(points):
    para_CL = np.cumsum(np.sqrt(
        np.sum((points[0:-1, :] - points[1:, :]) ** 2, axis=1)))
    para_CL = para_CL / para_CL.max()
    para_CL = np.concatenate((np.zeros(1), para_CL), axis=0)
    return para_CL
def Sample_AloneLine(point,normal,radius,N):
    coor_radius = np.random.uniform(low=-1, high=1, size=(N, 1))
    sample_pts = point + radius*normal * coor_radius
    return sample_pts,coor_radius
def Sample_Parameterize(parameterize, line, normal, radius, Nl, Nr, ForVisualization=False):
    if ForVisualization:
        coor_length = np.expand_dims(np.linspace(0,1,Nl),axis=1)
        coor_radius = np.expand_dims(np.linspace(-1,1,Nr),axis=1)
        coor_length = np.reshape(np.tile(coor_length,(Nr,1)), (Nl*Nr,1))
        coor_radius = np.reshape(np.tile(coor_radius,(1,Nl)), (Nl*Nr,1))
    else:
        coor_length = np.random.uniform(low=0, high=1, size=(Nl*Nr,1))
        coor_radius = np.random.uniform(low=-1, high=1, size=(Nl*Nr,1))

    sample_line = np.concatenate((np.interp(coor_length,parameterize,line[:,0]),
                                 np.interp(coor_length,parameterize,line[:,1])),axis=1)
    sample_nml = np.concatenate((np.interp(coor_length,parameterize,normal[:,0]),
                                 np.interp(coor_length,parameterize,normal[:,1])),axis=1)
    sample_rad = np.interp(coor_length,parameterize,radius)

    sample_pts = sample_line + sample_rad*sample_nml * coor_radius
    coor_radius[coor_radius==0] = 1
    sample_bdr = sample_line + sample_rad*sample_nml * coor_radius/np.abs(coor_radius)

    return sample_pts,sample_bdr,sample_nml

class SAMPLER2D_VESSEL:
    def __init__(self, pts_CL, radius_CL, smoother_CL,
                 time, visgrid=[20,10,5], device='cpu'):
        self.Nl, self.Nr, self.Nt = visgrid[0], visgrid[1], visgrid[2]
        self.device = device
        self.time = np.array(time)        # [Start, VisStart, VisEnd, End]

        self.pts_CL, self.normal_CL, self.radius_CL = \
            compute_smoothCL(pts_CL, radius_CL, smoother_CL)

        self.para_CL = parameterize_CL(self.pts_CL)
        self.vec_in = normalize_byrow(self.pts_CL[1:2, :] - self.pts_CL[0:1, :])
        self.vec_out = normalize_byrow(self.pts_CL[-1:, :] - self.pts_CL[-2:-1, :])
        vis_pts,vis_bdr,_ = Sample_Parameterize(
            self.para_CL, self.pts_CL, self.normal_CL, self.radius_CL,
            self.Nl, self.Nr,ForVisualization=True
        )
        vis_time = np.expand_dims(np.linspace(self.time[1], self.time[2], self.Nt), axis=1)
        vis_bdr, _ = pair_domainVStime(vis_bdr, vis_time, self.Nl, self.Nr, self.Nt)
        vis_pts, vis_time = pair_domainVStime(vis_pts, vis_time,self.Nl, self.Nr, self.Nt)
        self.vis_bdr = self.data_warper(vis_bdr)
        self.vis_pts = self.data_warper(vis_pts)
        self.vis_time = self.data_warper(vis_time)

    def sample_domain(self,N, t=None):
        T0,T3 = self.time[0],self.time[3]

        domain_t = np.random.uniform(low=T0, high=T3, size=(N, 1))
        domain_t = domain_t * 0 + t if t is not None else domain_t
        domain_xy = Sample_Parameterize(
            self.para_CL, self.pts_CL, self.normal_CL, self.radius_CL,
            N, 1, ForVisualization=False
        )[0]

        Xdomain = np.concatenate((domain_xy, domain_t), axis=1)
        return self.data_warper(Xdomain)
    def sample_wall(self,N, t=None):
        T0,T3 = self.time[0],self.time[3]

        boundary_t = np.random.uniform(low=T0, high=T3, size=(N, 1))
        boundary_t = boundary_t * 0 + t if t is not None else boundary_t
        boundary_xy,boundary_nml = Sample_Parameterize(
            self.para_CL, self.pts_CL, self.normal_CL, self.radius_CL,
            N, 1, ForVisualization=False
        )[1:3]

        XbdrWALL = np.concatenate((boundary_xy, boundary_t), axis=1)
        UbdrWALL = np.zeros((N,2))
        NbdrWALL = boundary_nml
        return self.data_warper(XbdrWALL), self.data_warper(UbdrWALL), self.data_warper(NbdrWALL)
    def sample_inlet(self, N, t=None):
        T0, T3 = self.time[0], self.time[3]

        bdrIN_t = np.random.uniform(low=T0, high=T3, size=(N, 1))
        bdrIN_t = bdrIN_t * 0 + t if t is not None else bdrIN_t
        bdrIN_xy, coor_radius = Sample_AloneLine(
            self.pts_CL[0, :], self.normal_CL[0, :], self.radius_CL[0], N
        )
        bdrIN_vu = Velocity * (1 - coor_radius ** 2) * self.vec_in

        XbdrIN = np.concatenate((bdrIN_xy, bdrIN_t), axis=1)
        UbdrIN = bdrIN_vu
        NbdrIN = np.tile(self.vec_in, (N, 1))
        return self.data_warper(XbdrIN), self.data_warper(UbdrIN), self.data_warper(NbdrIN)
    def sample_outlet(self,N, t=None):
        T0,T3 = self.time[0],self.time[3]

        bdrOUT_t = np.random.uniform(low=T0, high=T3, size=(N, 1))
        bdrOUT_t = bdrOUT_t * 0 + t if t is not None else bdrOUT_t
        bdrOUT_xy,coor_radius = Sample_AloneLine(
            self.pts_CL[-1,:], self.normal_CL[-1,:], self.radius_CL[-1],N
        )
        bdrOUT_vu = Velocity * (1 - coor_radius**2) * self.vec_out

        XbdrOUT = np.concatenate((bdrOUT_xy,bdrOUT_t),axis=1)
        UbdrOUT = bdrOUT_vu
        NbdrOUT = np.tile(self.vec_out,(N, 1))
        return self.data_warper(XbdrOUT), self.data_warper(UbdrOUT), self.data_warper(NbdrOUT)

    def visualize_sampled_domain(self):
        T1,T2 = self.time[1],self.time[2]
        N = 1000
        Nt = 5
        N_step = 10
        tstep=(T2-T1)/(Nt-1)
        plt.figure(figsize=(4, 10), dpi=600)
        for i in range(Nt):
            time = tstep*i
            plt.subplot(5, 1, i+1)
            plt.gca().axis('equal')
            dom = self.sample_domain(N, time).cpu().detach().numpy()
            bdr = torch.concat((
                self.sample_wall(N, time)[0],
                self.sample_outlet(N, time)[0]
            ),dim=0).cpu().detach().numpy()

            plt.scatter(dom[:, 0], dom[:, 1], s=1/10, c='b')
            plt.scatter(bdr[:, 0], bdr[:, 1], s=1/10, c='r')

        plt.show()

    def visualize(self,Nu=None,Np=None,Ns=None, function=None):
        N_step = 10
        X_reference = torch.concat((
            self.vis_pts,self.vis_time
        ),dim=1)

        if Ns is not None:
            D_current = Ns(X_reference)
            X_current = torch.concat((
                X_reference[:,0:1]+D_current[:,0:1],
                X_reference[:,1:2]+D_current[:,1:2],
                X_reference[:,-1:]
            ), dim=1)
        else:
            X_current = X_reference
        U_current = Nu(X_current)
        P_current = Np(X_current)

        if function is not None:
            U_current = torch.concat((
                function(X_current, LOSS_TYPE = 'PDE')[0],
                function(X_current, LOSS_TYPE = 'PDE')[0]
            ),dim=1)
            U_current[0,:] = 0*U_current[0,:]

        X_current_np = np.reshape(X_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl,3))
        U_current_np = np.reshape(U_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl,2))
        P_current_np = np.reshape(P_current.detach().cpu().numpy(),(self.Nt,self.Nr,self.Nl))
        Color = np.sqrt(np.sum(U_current_np ** 2, axis=-1))
        ColorMax, ColorMin = Color.max(), Color.min()
        print("Visualization range : {},{}".format(ColorMax, ColorMin))

        if self.Nt < 20:
            self.fig = plt.figure(figsize=(4, 10), dpi=600)
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
            plt.quiver(
                X_current_np[i, ::N_step, ::N_step, 0],
                X_current_np[i, ::N_step, ::N_step, 1],
                U_current_np[i, ::N_step, ::N_step, 0]/Velocity,
                U_current_np[i, ::N_step, ::N_step, 1]/Velocity, scale=Velocity
            )
            # plt.streamplot(
            #     X_current_np[i, ::N_step, ::N_step, 0],
            #     X_current_np[i, ::N_step, ::N_step, 1],
            #     U_current_np[i, ::N_step, ::N_step, 0],
            #     U_current_np[i, ::N_step, ::N_step, 1],
            #     # start_points=stream_points, density=[0.5, 1])
            #     density=0.6, color='k', linewidth=Color[i, ::N_step, ::N_step]/Velocity
            # )

            # # Plot for vessel boundary
            # plt.plot(X_current_np[i, :, 0, 0], X_current_np[i, :, 0, 1], c='r')
            # plt.plot(X_current_np[i, :, -1, 0], X_current_np[i, :, -1, 1], c='r')
            # plt.scatter(X_current_np[i, :, 0, 0], X_current_np[i, :, 0, 1], s=1, c='b')
            # plt.scatter(X_current_np[i, :, -1, 0], X_current_np[i, :, -1, 1], s=1, c='b')

            if self.Nt >= 20:
                plt.axis('off')
                self.fig.savefig('frame/f_{}_{}.png'.format(0, i))
                plt.close(self.fig)
        if self.Nt < 20:
            plt.show()
    def data_warper(self, data):
        return ToTensor(data,self.device)
    def update_time(self, time_bound):
        self.time[-1] = time_bound
        print("[S2S] SAMPLER time limit updated to {:2.2f}".format(time_bound))