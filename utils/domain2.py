import numpy as np
import matplotlib.pyplot as plt
def car2pol(y,x):
    return np.sqrt(x**2 + y**2),np.arctan2(y, x)
def rotate_via_numpy(x, y, radians):
    # Rotating in counter-clockwise order
    radians = np.expand_dims(radians,axis=-1)
    Xtran = np.concatenate((x,y),axis=-1)
    Xtran = np.expand_dims(Xtran,axis=-2)

    c, s = np.cos(radians), np.sin(radians)
    Rtran = np.concatenate((
        np.concatenate(( c, s),axis=-1),
        np.concatenate((-s, c),axis=-1)),axis=-2
    )
    Y = np.matmul(Xtran,Rtran)
    if len(Y.shape)==4:
        return Y[:,:,0,0:1],Y[:,:,0,1:2]
    elif len(Y.shape)==3:
        return Y[:,0,0:1],Y[:,0,1:2]

class ELLIPSE:
    def __init__(
            self,
            center_ref=[0.5,0],
            size=[0.06,0.03],
            period = [1,1],
            range = [0,0],
            time = [0,0,1,1]
    ):
        self.center_ref = center_ref
        self.size = size
        self.period = period #[rotate_period,translation_period]
        self.range = range #translation range [range for x,range for y]
        self.time = np.array(time)  # [Start, VisStart, VisEnd, End]

    def inside_ellipse(self,X):
        x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        # The rotated ellipse in polar coordinate form
        #   "-" indicates clockwise
        #   "+" indicates counter-clockwise
        center_cur_x = self.center_ref[0] + self.range[0] * np.sin(2*np.pi*t / self.period[1])
        center_cur_y = self.center_ref[1] + self.range[1] * np.sin(2*np.pi*t / self.period[1])
        
        x_origin = x-center_cur_x
        y_origin = y-center_cur_y
        rho,theta = car2pol(y_origin, x_origin)
        Ebdr_x_origin, Ebdr_y_origin,_,_ = self.sample_boundary_origin(theta, t)
        Ebdr_rho = np.sqrt(Ebdr_x_origin**2 + Ebdr_y_origin**2)

        return Ebdr_rho[:,0] >= rho[:,0]

    def sample_domain(self, N, t=None):
        T0,T1,T2,T3 = self.time[0],self.time[1],self.time[2],self.time[3]

        domain_t = np.random.uniform(low=T0, high=T3, size=(N, 1))
        if t is not None:
            domain_t = domain_t*0 + t
        domain_theta = np.random.uniform(low=0, high=2*np.pi, size=(N, 1))
        domain_rho = np.random.uniform(low=0, high=1, size=(N, 1))

        center_cur_x = self.center_ref[0] + self.range[0] * np.sin(2 * np.pi * domain_t / self.period[1])
        center_cur_y = self.center_ref[1] + self.range[1] * np.sin(2 * np.pi * domain_t / self.period[1])

        Edom_x_origin, Edom_y_origin = self.sample_domain_origin(domain_theta, domain_t, domain_rho)

        domain_x = center_cur_x + Edom_x_origin
        domain_y = center_cur_y + Edom_y_origin
        obj_dom = np.concatenate((
            domain_x, domain_y, domain_t
        ), axis=-1)
        return obj_dom
    def sample_domain_origin(self, domain_theta, domain_t, domain_rho):
        domain_psi = 2*np.pi * (domain_t/self.period[0])
        X_a = self.size[0] * np.cos(domain_theta-domain_psi) * domain_rho
        X_b = self.size[1] * np.sin(domain_theta-domain_psi) * domain_rho
        X_x,X_y = rotate_via_numpy(X_a, X_b, domain_psi)
        return X_x,X_y
    def visualize_sampled_domain(self,N, time):
        # plt.figure(figsize=(5, 10), dpi=150)
        obj_dom = self.sample_domain(N=N, t=time)
        plt.scatter(obj_dom[:, 0], obj_dom[:, 1], s=1, c='r')
        # plt.show()

    def generate_boundary(self, grid=[20,10,5]):
        T0,T1,T2,T3 = self.time[0],self.time[1],self.time[2],self.time[3]

        Nl, Nr, Nt = grid[0], grid[1], grid[2]
        boundary_t = np.tile(np.expand_dims(np.linspace(T1, T2, Nt), axis=(1,2)),(1,Nl,1))
        boundary_theta = np.tile(np.expand_dims(np.linspace(0, 2*np.pi, Nl), axis=(0,2)),(Nt,1,1))

        center_cur_x = self.center_ref[0] + self.range[0] * np.sin(2*np.pi*boundary_t / self.period[1])
        center_cur_y = self.center_ref[1] + self.range[1] * np.sin(2*np.pi*boundary_t / self.period[1])

        Ebdr_x_origin,Ebdr_y_origin,Ebdr_v_origin,Ebdr_u_origin = self.sample_boundary_origin(boundary_theta, boundary_t)

        boundary_x = center_cur_x + Ebdr_x_origin
        boundary_y = center_cur_y + Ebdr_y_origin
        obj_bdr = np.concatenate((
            boundary_x,boundary_y,boundary_t
        ),axis=-1)

        C0 = self.range[0] *2*np.pi / self.period[1]
        C1 = self.range[1] *2*np.pi / self.period[1]
        boundary_v = Ebdr_v_origin + C0 * np.sin(2*np.pi*boundary_t / self.period[1])
        boundary_u = Ebdr_u_origin + C1 * np.sin(2*np.pi*boundary_t / self.period[1])
        obj_bdr_vel = np.concatenate((
            boundary_v,boundary_u
        ),axis=-1)


        # Nt = 5
        # plt.figure(figsize=(5, 10), dpi=150)
        # for i in range(Nt):
        #     plt.subplot(5, 1, i+1)
        #     plt.gca().axis('equal')
        #     plt.plot(obj_bdr[i,:,0], obj_bdr[i,:,1], 'r')
        #     plt.quiver(
        #         obj_bdr[i,:,0],
        #         obj_bdr[i,:,1],
        #         obj_bdr_vel[i,:,0],
        #         obj_bdr_vel[i,:,1], scale=1
        #     )
        # plt.show()
        return obj_bdr,obj_bdr_vel
    def sample_boundary(self, N, t=None):
        T0,T1,T2,T3 = self.time[0],self.time[1],self.time[2],self.time[3]

        boundary_t = np.random.uniform(low=T0, high=T3, size=(N, 1))
        if t is not None:
            boundary_t = boundary_t*0 + t
        boundary_theta = np.random.uniform(low=0, high=2*np.pi, size=(N, 1))
        center_cur_x = self.center_ref[0] + self.range[0] * np.sin(2 * np.pi * boundary_t / self.period[1])
        center_cur_y = self.center_ref[1] + self.range[1] * np.sin(2 * np.pi * boundary_t / self.period[1])

        Ebdr_x_origin, Ebdr_y_origin, Ebdr_v_origin, Ebdr_u_origin = self.sample_boundary_origin(boundary_theta,
                                                                                                   boundary_t)
        boundary_x = center_cur_x + Ebdr_x_origin
        boundary_y = center_cur_y + Ebdr_y_origin
        obj_bdr = np.concatenate((
            boundary_x,boundary_y,boundary_t
        ),axis=-1)

        C0 = self.range[0] *2*np.pi / self.period[1]
        C1 = self.range[1] *2*np.pi / self.period[1]
        boundary_v = Ebdr_v_origin + C0 * np.sin(2*np.pi*boundary_t / self.period[1])
        boundary_u = Ebdr_u_origin + C1 * np.sin(2*np.pi*boundary_t / self.period[1])
        obj_bdr_vel = np.concatenate((
            boundary_v,boundary_u
        ),axis=-1)
        return obj_bdr,obj_bdr_vel

    def sample_boundary_origin(self, boundary_theta, boundary_t):
        boundary_psi = 2*np.pi * (boundary_t/self.period[0])
        X_a = self.size[0] * np.cos(boundary_theta-boundary_psi)
        X_b = self.size[1] * np.sin(boundary_theta-boundary_psi)
        X_x,X_y = rotate_via_numpy(X_a, X_b, boundary_psi)

        X_rho, X_theta = car2pol(X_y, X_x)
        V_x = - np.sin(X_theta) * X_rho
        V_y =   np.cos(X_theta) * X_rho
        return X_x,X_y,V_x,V_y