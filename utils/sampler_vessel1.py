import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import ToTensor

Velocity = 10
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
                 Time=np.array([0,0,1,1]), visgrid=[40,5],device='cpu'):
        super(SAMPLER_VESSEL, self).__init__()
        self.Nl, self.Nt = visgrid[0],visgrid[1]
        self.device = device
        self.Connect = Connect
        self.Points = Points
        self.Radius = Radius
        self.Length,self.Radius_z = compute_preprocess(Connect,Points,Radius)
        self.Time = Time
        self.N_segment = Connect.shape[0]
        self.vis_points,self.vis_radius,self.vis_time,self.vis_z_Pseg \
            = generate_vessel_grid(visgrid,Connect,Points,Radius,self.Length,Time)
    def sample_domain(self,N, t=None):
        vesseltree_domain = []
        vesseltree_r = []
        vesseltree_r_z = []
        for j in range(self.N_segment):
            length,_,_,R0,R1 = extract_segmentINFO(self.Connect,self.Points,self.Radius,j)

            length = self.Length[j]
            this_z = np.random.uniform(low=0, high=length, size=(N, 1))
            this_r = (length-this_z)/length*R0 + this_z/length*R1
            this_t = np.random.uniform(low=self.Time[0], high=self.Time[3], size=(N, 1))
            this_r_z = np.tile(self.Radius_z[j],(N,1))

            vesseltree_domain += [self.data_warper(np.concatenate((this_z,this_t),  axis=1))]
            vesseltree_r += [self.data_warper(this_r)]
            vesseltree_r_z += [self.data_warper(this_r_z)]
        return vesseltree_domain,vesseltree_r,vesseltree_r_z
    def sample_inlet(self,N, t=None):
        vesseltree_inlet = []
        for j in range(self.N_segment):
            this_z = np.random.uniform(low=0, high=0, size=(N, 1))
            this_t = np.random.uniform(low=self.Time[0], high=self.Time[3], size=(N, 1))
            vesseltree_inlet += [self.data_warper(np.concatenate((this_z, this_t), axis=1))]
            if j==0:
                inlet_velocity = self.data_warper(self.sample_inletV(this_t))
        return vesseltree_inlet,inlet_velocity
    def sample_outlet(self,N, t=None):
        vesseltree_outlet = []
        for j in range(self.N_segment):
            length = self.Length[j]
            this_z = np.random.uniform(low=length, high=length, size=(N, 1))
            this_t = np.random.uniform(low=self.Time[0], high=self.Time[3], size=(N, 1))
            vesseltree_outlet += [self.data_warper(np.concatenate((this_z, this_t), axis=1))]
        return vesseltree_outlet
    def sample_inletV(self,t):
        v = (1 - np.cos(2*np.pi*t)) / 2
        return 10 * v
    def data_warper(self, data):
        return ToTensor(data,self.device)
    def getConnect(self):
        return self.Connect
    def visualize(self,N_Q=None,N_P=None,N_R=None):
        VIDEOFLAG = (self.Nt > 20)
        self.fig = plt.figure(figsize=(4, 10), dpi=150) if ~VIDEOFLAG else None

        for k in range(self.Nt):
            for j in range(self.N_segment):
                X = self.data_warper(np.concatenate((
                    self.vis_time[j][k,:,:],self.vis_z_Pseg[j][k,:,:]
                ),axis=1))
                Q = N_Q(X)
                P = N_P(X)
                R = N_R(X)

                points = self.vis_points[j][k,:,:]
                flux = Q.detach().cpu().numpy()
                pressure = P.detach().cpu().numpy()
                radius = R.detach().cpu().numpy()+ self.vis_radius[j][k,:,:]
                boundary0 = compute_bdrpoint(points,radius)
                boundary1 = compute_bdrpoint(points,-radius)
                this_color = np.concatenate((flux,flux,flux),axis=1)
                ColorMax, ColorMin = 20,0

                this_grid = np.concatenate((
                    np.expand_dims(boundary0,axis=1),
                    np.expand_dims(points,axis=1),
                    np.expand_dims(boundary1,axis=1)
                ),axis=1)

                if VIDEOFLAG:
                    self.fig = plt.figure(figsize=(8, 4), dpi=150)
                else:
                    plt.subplot(self.Nt, 1, k + 1)

                plt.tight_layout()
                plt.gca().axis('equal')
                plt.gca().pcolormesh(
                    this_grid[:, :, 0],
                    this_grid[:, :, 1],
                    this_color,
                    vmin=ColorMin, vmax=ColorMax
                )

                if VIDEOFLAG:
                    plt.axis('off')
                    self.fig.savefig('frame/f_{}_{}.png'.format(0, j))
                    plt.close(self.fig)
        if ~VIDEOFLAG:
            plt.show()