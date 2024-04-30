import numpy as np
import matplotlib.pyplot as plt
def car2pol(y,x):
    return np.sqrt(x**2 + y**2),np.arctan2(y, x)
def rotate_via_numpy(x, y, radians):
    radians = np.expand_dims(radians,axis=-1)
    Xtran = np.concatenate((x,y),axis=-1)
    Xtran = np.expand_dims(Xtran,axis=-2)

    c, s = np.cos(radians), np.sin(radians)
    Rtran = np.concatenate((
        np.concatenate(( c,-s),axis=-1),
        np.concatenate(( s, c),axis=-1)),axis=-2
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
        Ebdr_x_origin, Ebdr_y_origin = self.generate_boundary_origin(theta, t)
        Ebdr_rho = np.sqrt(Ebdr_x_origin**2 + Ebdr_y_origin**2)

        return Ebdr_rho[:,0] >= rho[:,0]

    def generate_domain(self, N, t=None):
        T0,T1,T2,T3 = self.time[0],self.time[1],self.time[2],self.time[3]

        domain_t = np.random.uniform(low=T0, high=T3, size=(N, 1))
        if t is not None:
            domain_t = domain_t*0 + t
        domain_theta = np.random.uniform(low=0, high=2*np.pi, size=(N, 1))
        domain_rho = np.random.uniform(low=0, high=1, size=(N, 1))

        center_cur_x = self.center_ref[0] + self.range[0] * np.sin(2 * np.pi * domain_t / self.period[1])
        center_cur_y = self.center_ref[1] + self.range[1] * np.sin(2 * np.pi * domain_t / self.period[1])

        Edom_x_origin, Edom_y_origin = self.generate_domain_origin(domain_theta, domain_t, domain_rho)

        domain_x = center_cur_x + Edom_x_origin
        domain_y = center_cur_y + Edom_y_origin
        obj_dom = np.concatenate((
            domain_x, domain_y, domain_t
        ), axis=-1)
        return obj_dom
    def visualize_sampled_domain(self,domain):
        plt.figure(figsize=(5, 10), dpi=150)

        obj_dom = self.generate_domain(N=1000, t=0)
        plt.subplot(5, 1, 1)
        plt.scatter(obj_dom[:, 0], obj_dom[:, 1], s=1, c='b')
        plt.gca().set_xlim([-0.1*domain[0],1.1*domain[0]])
        plt.gca().set_ylim([-0.6*domain[1],0.6*domain[1]])

        obj_dom = self.generate_domain(N=1000, t=0.125)
        plt.subplot(5, 1, 2)
        plt.scatter(obj_dom[:, 0], obj_dom[:, 1], s=1, c='b')
        plt.gca().set_xlim([-0.1*domain[0],1.1*domain[0]])
        plt.gca().set_ylim([-0.6*domain[1],0.6*domain[1]])

        obj_dom = self.generate_domain(N=1000, t=0.250)
        plt.subplot(5, 1, 3)
        plt.scatter(obj_dom[:, 0], obj_dom[:, 1], s=1, c='b')
        plt.gca().set_xlim([-0.1*domain[0],1.1*domain[0]])
        plt.gca().set_ylim([-0.6*domain[1],0.6*domain[1]])

        obj_dom = self.generate_domain(N=1000, t=0.375)
        plt.subplot(5, 1, 4)
        plt.scatter(obj_dom[:, 0], obj_dom[:, 1], s=1, c='b')
        plt.gca().set_xlim([-0.1*domain[0],1.1*domain[0]])
        plt.gca().set_ylim([-0.6*domain[1],0.6*domain[1]])

        obj_dom = self.generate_domain(N=1000, t=0.500)
        plt.subplot(5, 1, 5)
        plt.scatter(obj_dom[:, 0], obj_dom[:, 1], s=1, c='b')
        plt.gca().set_xlim([-0.1*domain[0],1.1*domain[0]])
        plt.gca().set_ylim([-0.6*domain[1],0.6*domain[1]])

        plt.show()
    def generate_domain_origin(self, domain_theta, domain_t, domain_rho):
        domain_psi = 2*np.pi * (domain_t/self.period[0])
        X_a = self.size[0] * np.sin(domain_theta-domain_psi) * domain_rho
        X_b = self.size[1] * np.cos(domain_theta-domain_psi) * domain_rho
        X_x,X_y = rotate_via_numpy(X_a, X_b, domain_psi)
        return X_x,X_y

    def generate_boundary(self, grid=[20,10,5]):
        T0,T1,T2,T3 = self.time[0],self.time[1],self.time[2],self.time[3]

        Nl, Nr, Nt = grid[0], grid[1], grid[2]
        boundary_t = np.tile(np.expand_dims(np.linspace(T1, T2, Nt), axis=(1,2)),(1,Nl,1))
        boundary_theta = np.tile(np.expand_dims(np.linspace(0, 2*np.pi, Nl), axis=(0,2)),(Nt,1,1))

        center_cur_x = self.center_ref[0] + self.range[0] * np.sin(2*np.pi*boundary_t / self.period[1])
        center_cur_y = self.center_ref[1] + self.range[1] * np.sin(2*np.pi*boundary_t / self.period[1])

        Ebdr_x_origin,Ebdr_y_origin = self.generate_boundary_origin(boundary_theta, boundary_t)

        boundary_x = center_cur_x + Ebdr_x_origin
        boundary_y = center_cur_y + Ebdr_y_origin
        obj_bdr = np.concatenate((
            boundary_x,boundary_y
        ),axis=-1)
        return obj_bdr

    def generate_boundary_origin(self, boundary_theta, boundary_t):
        boundary_psi = 2*np.pi * (boundary_t/self.period[0])
        X_a = self.size[0] * np.sin(boundary_theta-boundary_psi)
        X_b = self.size[1] * np.cos(boundary_theta-boundary_psi)
        X_x,X_y = rotate_via_numpy(X_a, X_b, boundary_psi)
        return X_x,X_y
