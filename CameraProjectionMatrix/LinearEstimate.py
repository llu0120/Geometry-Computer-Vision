'''
Linear Estimation of the Camera Projection Matrix by Direct Linear Transformation (DLT) 
'''
from utility import *

class LinearEstimate():
    def Homogenize(self, x):
        # converts points from inhomogeneous to homogeneous coordinates
        return np.vstack((x,np.ones((1,x.shape[1]))))

    def Dehomogenize(self, x):
        # converts points from homogeneous to inhomogeneous coordinates
        return x[:-1]/x[-1]
    
    
    def Normalize(self, pts):
        # data normalization of n dimensional pts
        #
        # Input:
        #    pts - is in inhomogeneous coordinates
        # Outputs:
        #    pts - data normalized points
        #    T - corresponding transformation matrix
        """your code here"""
        ndim = len(pts)
        mean = np.mean(pts, axis=1)
        var = np.var(pts, axis=1)
        total_var = np.sum(var)
    
        #2D
        if (ndim == 2):
            s = np.sqrt(2/total_var)
            Hs = np.array([[s, 0, -mean[0]*s],
                           [0, s, -mean[1]*s],
                           [0, 0,  1]])
        #3D
        if (ndim == 3):
            s = np.sqrt(3/total_var)
            Hs = np.array([[s, 0, 0, -mean[0]*s],
                           [0, s, 0, -mean[1]*s],
                           [0, 0, s, -mean[2]*s],
                           [0, 0, 0, 1]])
        pts = np.dot(Hs, self.Homogenize(pts))
        T = Hs
        return pts, T
    
    def leftNull(self, x): 
        #Input: 
        #    x = (x1, x2, x3, ..., xn).T  - any vector    
        n = len(x)
        x_norm = np.linalg.norm(x)
        e1 = np.zeros((len(x),1))
        e1[0] = 1
        v = x + np.sign(x[0]) * x_norm * e1 #n x 1
        Hv = np.eye(n) - (2 * (np.dot(v, np.transpose(v)) / np.dot(np.transpose(v), v)))
        x_leftNull = Hv[1:, :]
        return x_leftNull

    def ComputeCost(self, P, x, X):
        # Inputs:
        #    x - 2D inhomogeneous image points
        #    X - 3D inhomogeneous scene points
        #
        # Output:
        #    cost - Total reprojection error
        n = x.shape[1]
        covarx = np.eye(2*n)
        
        """your code here"""
        x_hat = np.dot(P, self.Homogenize(X))
        epslon = x - self.Dehomogenize(x_hat)
        
        epslon = np.reshape(epslon, (2*n, 1), 'F') #2n x 1
        epslonT = np.transpose(epslon) #1 x 2n
        cost = np.dot(np.dot(epslonT, np.linalg.inv(covarx)), epslon)
        
        return cost
    
    def DLT(self, x, X, normalize=True):
        # Inputs:
        #    x - 2D inhomogeneous image points
        #    X - 3D inhomogeneous scene points
        #    normalize - if True, apply data normalization to x and X
        #
        # Output:
        #    P - the (3x4) DLT estimate of the camera projection matrix
        P = np.eye(3,4)+np.random.randn(3,4)/10
            
        # data normalization
        if normalize:
            x, T = self.Normalize(x)
            X, U = self.Normalize(X)
        else:
            x = self.Homogenize(x)
            X = self.Homogenize(X)
        
        """your code here"""
        numPoint = len(x[0])
        A = np.zeros((2*numPoint, 12))
        for i in range(numPoint):
            point = x[:, i:i+1]
            POINT = X[:, i:i+1]
            x_leftNull = self.leftNull(point)
        
            A[2*i:2*i+2, :] = np.kron(x_leftNull, np.transpose(POINT))
       
            u, s, v = np.linalg.svd(A)
            v = v[11:,:]
        
            P = np.reshape(v, np.shape(P))
        
    
        # data denormalize
        if normalize:
            P = np.linalg.inv(T) @ P @ U
            
        return P
    
    
