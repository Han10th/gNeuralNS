import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import ToTensor

velocity_max = 10
def extract_segmentIDX(Connect,j):
    if j==0:
        startID4point = 0
        endID4point = 1
    else:
        startID4point = np.where(Connect[j,:])[0][0]+1
        endID4point = j+1
    return startID4point,endID4point
def extract_segmentINFO(Connect,Points,Radius,j):
    startID4point,endID4point = extract_segmentIDX(Connect, j)
    P0 = Points[startID4point,:]
    P1 = Points[endID4point,:]
    R0 = Radius[j,0]
    R1 = Radius[j,1]
    length = np.linalg.norm(P0 - P1)
    return length,P0,P1,R0,R1
def compute_preprocess(Connect,Points,Radius):
    """
        The vessel tree of N segments is recorded by
        vessel_connect (connectivity, NxN) and vessel_pts (key points, (N+1)x3).
        The first segment is (vessel_pts[0,:],vessel_pts[1,:])
        The j-th segments are (vessel_pts[i+1,:],vessel_pts[j+1,:]) where vessel_connect[i,j]==1
    """
    N_segment = Connect.shape[0]
    Length = np.zeros(N_segment)
    for j in range(N_segment):
        length,_,_,_,_ = extract_segmentINFO(Connect,Points,Radius,j)
        Length[j] = length
    Radius_z = (Radius[:,1]-Radius[:,0]) / (Length[:])
    return Length,Radius_z
def compute_bdrpoint(points,radius):
    tangent = points[-1,:] - points[0,:]
    tangent = tangent / np.linalg.norm(tangent)
    vector = np.array([0,0,1])
    if np.linalg.norm(np.cross(tangent,vector)) == 0:
        vector = np.array([0,1,0])
    normal = np.cross(tangent,vector)
    normal = normal / np.linalg.norm(normal)
    # binormal = np.cross(tangent,normal)
    # binormal = binormal / np.linalg.norm(binormal)
    return points + radius*normal
def generate_vessel_grid(grid,Connect,Points,Radius,Length,Time):
    N_segment = Connect.shape[0]
    Nl,Nt = grid[0],grid[1],

    vis_z = []
    vis_r = []
    vis_pts=[]
    vis_time = []
    for j in range(N_segment):
        length,P0,P1,R0,R1 = extract_segmentINFO(Connect,Points,Radius,j)
        thisNl = np.int16(length*Nl)

        this_z = np.expand_dims(np.linspace(0,length,thisNl), axis=(0,2))
        this_r = (length-this_z)/length*R0 + this_z/length*R1
        this_pts = np.concatenate((
            np.expand_dims(np.linspace(P0[0], P1[0], thisNl), axis=(0,2)),
            np.expand_dims(np.linspace(P0[1], P1[1], thisNl), axis=(0,2)),
            np.expand_dims(np.linspace(P0[2], P1[2], thisNl), axis=(0,2)),
        ),axis=-1)
        this_t = np.expand_dims(np.linspace(Time[1], Time[2], Nt), axis=(1,2))

        vis_z += [np.tile(this_z, (Nt, 1, 1))]
        vis_r += [np.tile(this_r, (Nt, 1, 1))]
        vis_pts += [np.tile(this_pts, (Nt, 1, 1))]
        vis_time += [np.tile(this_t, (1, thisNl,1))]

    return vis_pts,vis_r,vis_time,vis_z


class SAMPLER_VESSEL:
    def __init__(self,
                 Connect,
                 Points,
                 Radius,
                 Time=[0,0,1,1], visgrid=[40,5],device='cpu',P_ext=0,beta=1):
        super(SAMPLER_VESSEL, self).__init__()
        self.Nl, self.Nt = visgrid[0],visgrid[1]
        self.device = device
        self.Connect = Connect
        self.Points = Points
        self.Radius = Radius
        self.Length,self.Radius_z = compute_preprocess(Connect,Points,Radius)
        self.Time = np.array(Time)
        self.P_ext = P_ext
        self.beta = beta
        self.N_segment = Connect.shape[0]
        self.vis_points,self.vis_radius,self.vis_time,self.vis_z_Pseg \
            = generate_vessel_grid(visgrid,Connect,Points,Radius,self.Length,Time)
    def sample_time(self,N):
        return np.random.uniform(low=self.Time[0], high=self.Time[3], size=(N, 1))
    def sample_domain(self,N, t=None):
        vesseltree_domain = []
        vesseltree_r = []
        vesseltree_r_z = []
        for j in range(self.N_segment):
            length,_,_,R0,R1 = extract_segmentINFO(self.Connect,self.Points,self.Radius,j)

            length = self.Length[j]
            this_z = np.random.uniform(low=0, high=length, size=(N, 1))
            this_r = (length-this_z)/length*R0 + this_z/length*R1
            if t is None:
                this_t = np.random.uniform(low=self.Time[0], high=self.Time[3], size=(N, 1))
            else:
                this_t = t
            this_r_z = np.tile(self.Radius_z[j],(N,1))

            vesseltree_domain += [self.data_warper(np.concatenate((this_z,this_t),  axis=1))]
            vesseltree_r += [self.data_warper(this_r)]
            vesseltree_r_z += [self.data_warper(this_r_z)]
        return vesseltree_domain,vesseltree_r,vesseltree_r_z
    def sample_inlet(self,N, t=None):
        vesseltree_inlet = []
        for j in range(self.N_segment):
            this_z = np.random.uniform(low=0, high=0, size=(N, 1))
            if t is None:
                this_t = np.random.uniform(low=self.Time[0], high=self.Time[3], size=(N, 1))
            else:
                this_t = t
            vesseltree_inlet += [self.data_warper(np.concatenate((this_z, this_t), axis=1))]
            if j==0:
                inlet_velocity = self.data_warper(self.sample_inletV(this_t))
        return vesseltree_inlet,inlet_velocity
    def sample_outlet(self,N, t=None):
        vesseltree_outlet = []
        for j in range(self.N_segment):
            length = self.Length[j]
            this_z = np.random.uniform(low=length, high=length, size=(N, 1))
            if t is None:
                this_t = np.random.uniform(low=self.Time[0], high=self.Time[3], size=(N, 1))
            else:
                this_t = t
            vesseltree_outlet += [self.data_warper(np.concatenate((this_z, this_t), axis=1))]
        return vesseltree_outlet
    def sample_inletV(self,t):
        v = (1 - np.cos(20*np.pi*t)) / 2 * ((t%0.2)<0.1)
        return velocity_max * v
    def sample_demo(self,N, t=None):
        vesseltree_demo = []
        vesseltree_velocity = []


        for j in range(self.N_segment):
            length,_,_,R0,R1 = extract_segmentINFO(self.Connect,self.Points,self.Radius,j)

            length = self.Length[j]
            this_z = np.random.uniform(low=0, high=length, size=(N, 1))
            this_r = (length-this_z)/length*R0 + this_z/length*R1
            if t is None:
                this_t = np.random.uniform(low=self.Time[0], high=self.Time[3], size=(N, 1))
            else:
                this_t = t
            this_r_z = np.tile(self.Radius_z[j],(N,1))

            numj = 1 if j!=0 else 0
            vesseltree_demo += [self.data_warper(np.concatenate((this_z, this_t), axis=1))]
            vesseltree_velocity += [self.data_warper(self.sample_velocity_demo(this_z/length+numj,this_t))]
        # plt.figure(figsize=(8, 8), dpi=150)
        # plt.plot(np.concatenate((
        #              this_z/length,this_z/length+1
        #          ),axis=0),
        #          np.concatenate((
        #              self.sample_velocity_demo(this_z/length + 0, this_t),
        #              self.sample_velocity_demo(this_z/length + 1, this_t)
        #          ),axis=0)
        #          ,'r.')
        # plt.show()
        return vesseltree_demo, vesseltree_velocity
    def sample_velocity_demo(self,z,t):
        z=z/2
        x= t-0.2*z
        v = (1-np.cos(20*np.pi*x))/2 * ((x%0.4)<0.1) - (2.25-(2.5-10*(x%0.4))**2)*0.07 * ((x%0.4)>=0.1)
        factor = (((t%0.4)-0.4)**2) *3 + 0.52
        return velocity_max * v * factor
    def data_warper(self, data):
        return ToTensor(data,self.device)
    def getConnect(self):
        return self.Connect
    def visualize(self,N_Q_list,N_P_list):
        VIDEOFLAG = (self.Nt > 20)
        self.fig = plt.figure(figsize=(8, 10), dpi=150) if ~VIDEOFLAG else None

        for k in range(self.Nt):

            if VIDEOFLAG:
                self.fig = plt.figure(figsize=(8, 4), dpi=150)

            for j in range(self.N_segment):
                X = self.data_warper(np.concatenate((
                    self.vis_z_Pseg[j][k,:,:],self.vis_time[j][k,:,:]
                ),axis=1))
                Q = N_Q_list[j](X)
                P = N_P_list[j](X)

                points = self.vis_points[j][k,:,:]
                radius0 = self.vis_radius[j][k,:,:]
                flux = Q.detach().cpu().numpy()/radius0
                pressure = P.detach().cpu().numpy()
                radius = 50*np.sqrt(np.pi)*radius0*radius0/self.beta*(pressure-self.P_ext) + radius0
                # radius = radius0
                boundary0 = compute_bdrpoint(points,radius)
                boundary1 = compute_bdrpoint(points,-radius)
                this_color = np.concatenate((flux,flux,flux),axis=1)
                this_press = np.concatenate((pressure,pressure,pressure),axis=1)
                ColorMax, ColorMin = velocity_max,0

                this_grid = np.concatenate((
                    np.expand_dims(boundary0,axis=1),
                    np.expand_dims(points,axis=1),
                    np.expand_dims(boundary1,axis=1)
                ),axis=1)

                if VIDEOFLAG is not True:
                    plt.tight_layout()
                    plt.subplot(self.Nt, 2, 2*k + 1)
                    plt.gca().axis('equal')
                    plt.gca().pcolormesh(
                        this_grid[:, :, 0],
                        this_grid[:, :, 1],
                        this_color,
                        vmin=ColorMin, vmax=ColorMax/1.5
                    )
                    plt.subplot(self.Nt, 2, 2*k + 2)
                plt.gca().axis('equal')
                plt.gca().pcolormesh(
                    this_grid[:, :, 0],
                    this_grid[:, :, 1],
                    this_press,
                    vmin=-20, vmax=100
                )
                plt.gca().set_xlim([-5, 45])
                plt.gca().set_ylim([-12, 12])

            if VIDEOFLAG:
                plt.axis('off')
                self.fig.savefig('frame/f_{}_{}.png'.format(0, k))
                plt.close(self.fig)
        if ~VIDEOFLAG:
            plt.show()

    def update_time(self, time_bound):
        self.Time[-1] = time_bound
        print("[S2S] SAMPLER time limit updated to {:2.2f}".format(time_bound))