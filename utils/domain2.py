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
    def __init__(self, center_ref=[0.5,0], size=[0.06,0.03], period = [1,1], range = [0,0]):
        self.center_ref = center_ref
        self.size = size
        self.period = period #[rotate_period,translate_period]
        self.range = range

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
        Ebdr_x_origin, Ebdr_y_origin = self.ellipse_bdr_origin(theta, t)
        Ebdr_rho = np.sqrt(Ebdr_x_origin**2 + Ebdr_y_origin**2)

        return Ebdr_rho >= rho

    def generate_boundary(self, time, grid=[20,10,5]):
        Nl, Nr, Nt = grid[0], grid[1], grid[2]
        t = np.tile(np.expand_dims(np.linspace(time[0], time[1], Nt), axis=(1,2)),(1,Nl,1))
        theta = np.tile(np.expand_dims(np.linspace(0, 2*np.pi, Nl), axis=(0,2)),(Nt,1,1))

        center_cur_x = self.center_ref[0] + self.range[0] * np.sin(2*np.pi*t / self.period[1])
        center_cur_y = self.center_ref[1] + self.range[1] * np.sin(2*np.pi*t / self.period[1])

        Ebdr_x_origin,Ebdr_y_origin = self.ellipse_bdr_origin(theta, t)

        x = center_cur_x + Ebdr_x_origin
        y = center_cur_y + Ebdr_y_origin
        obj_bdr = np.concatenate((x,y),axis=-1)
        return obj_bdr

    def ellipse_bdr_origin(self, theta, t):
        psi = 2*np.pi * (t/self.period[0])
        X_a = self.size[0] * np.sin(theta-psi)
        X_b = self.size[1] * np.cos(theta-psi)
        X_x,X_y = rotate_via_numpy(X_a, X_b, psi)
        return X_x,X_y
