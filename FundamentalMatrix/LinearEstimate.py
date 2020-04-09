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
    
    def DLT(self, x1, x2, normalize=True):
        # Inputs:
        #    x1 - inhomogeneous inlier correspondences in image 1
        #    x2 - inhomogeneous inlier correspondences in image 2
        #    normalize - if True, apply data normalization to x1 and x2
        #
        # Outputs:
        #    F - the DLT estimate of the fundamental matrix  
        
        """your code here"""
        numPoint = len(x1[0])
        # data normalization
        if normalize:
            x1, T1 = self.Normalize(x1)
            x2, T2 = self.Normalize(x2)
            
        A = np.zeros((numPoint, 9))
        for i in range(numPoint):
            x_kron = np.kron(np.transpose(x2[:, i:i+1]), np.transpose(x1[:, i:i+1]))
            A[i:i+1, :] = x_kron
        
        U, S, VT = np.linalg.svd(A)
        f = VT[8:9, :]
        F = np.reshape(f, (3, 3))
        
        #Enforce Rank Constraint 
        U, S, VT = np.linalg.svd(F)
        S[-1] = 0 
        Diag_S = np.diag(S)
        
        F = U@Diag_S@VT
        
        # data denormalization
        if normalize:
            F = np.transpose(T2)@F@T1
            
        F = F/np.linalg.norm(F)
        
        return F
    
    
