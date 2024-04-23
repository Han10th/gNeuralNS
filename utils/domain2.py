import numpy as np
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
    return Y[:,:,0,0:1],Y[:,:,0,1:2]

class ELLIPSE:
    def __init__(self, ellipse_center=[0.5,0], ellipse_size=[0.06,0.03], rotating_period = 1):
        self.ellipse_center = ellipse_center
        self.ellipse_size = ellipse_size
        self.rotating_period = rotating_period

    def inside_ellipse(self,X):
        x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        # The rotated ellipse in polar coordinate form
        #   "-" indicates clockwise
        #   ”+“ indicates counter-clockwise
        vec_x = x-self.ellipse_center[0]
        vec_y = y-self.ellipse_center[1]
        rho,theta = car2pol(vec_y, vec_x)
        Ebdr_x,Ebdr_y = self.ellipse_bdr(theta, t)
        Ebdr_rho = np.sqrt(Ebdr_x**2 + Ebdr_y**2)

        return Ebdr_rho >= rho

    def generate_boundary(self, time, grid=[20,10,5]):
        Nl, Nr, Nt = grid[0], grid[1], grid[2]
        obj_t = np.tile(np.expand_dims(np.linspace(time[0], time[1], Nt), axis=(1,2)),(1,Nl,1))
        obj_theta = np.tile(np.expand_dims(np.linspace(0, 2*np.pi, Nl), axis=(0,2)),(Nt,1,1))

        Ebdr_x,Ebdr_y = self.ellipse_bdr(obj_theta, obj_t)

        x = Ebdr_x + self.ellipse_center[0]
        y = Ebdr_y + self.ellipse_center[1]
        obj_bdr = np.concatenate((x,y),axis=-1)
        return obj_bdr

    def ellipse_bdr(self, theta, t):
        psi = 2*np.pi * (t/self.rotating_period)
        X_a = self.ellipse_size[0] * np.sin(theta-psi)
        X_b = self.ellipse_size[1] * np.cos(theta-psi)
        X_x,X_y = rotate_via_numpy(X_a, X_b, psi)
        return X_x,X_y
