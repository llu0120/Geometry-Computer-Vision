from utility import *
"""
Linear Estimation by DLT
"""

class LinearEstimate():
    
    def Homogenize(self, x):
        # converts points from inhomogeneous to homogeneous coordinates
        return np.vstack((x,np.ones((1,x.shape[1]))))

    def Dehomogenize(self, x):
        # converts points from homogeneous to inhomogeneous coordinates
        return x[:-1]/x[-1]
    
    def computeError(self, pt1, pt2, H, T1, T2, normalize=True): #total error + correction
        # Inputs:
        #    pt1 - a 2D inhomogeneous image1 point
        #    pt2 - a 2D inhomogeneous image2 point
        #
        # Output:
        #    error - error
        h11, h12, h13 = H[0][0], H[0][1], H[0][2]
        h21, h22, h23 = H[1][0], H[1][1], H[1][2]
        h31, h32, h33 = H[2][0], H[2][1], H[2][2]
        
        xi = float(pt1[0])
        yi = float(pt1[1])
        xi_prime = float(pt2[0])
        yi_prime = float(pt2[1])
        J = np.array([[-h21+yi_prime*h31, -h22+yi_prime*h32, 0, xi*h31+yi*h32+h33],
                      [h11-xi_prime*h31, h12-xi_prime*h32, -(xi*h31+yi*h32+h33), 0]])
        JT = np.transpose(J)
        epslon = np.array([[-(xi*h21 + yi*h22 + h23) + yi_prime*(xi*h31 + yi*h32 + h33)],
                           [xi*h11 + yi*h12 + h13 - xi_prime*(xi*h31 + yi*h32 + h33)]])
        
        #error = np.transpose(epslon)@(J@JT)@epslon #This is Sampson error
    
        lamda = np.dot(np.linalg.inv(np.dot(J, JT)), (-epslon))
        error_tmp = np.dot(JT, lamda) # 4 x 1 
        
        #Sampson correction 
        error_x1, error_y1 = float(error_tmp[0]), float(error_tmp[1])
        error_x2, error_y2 = float(error_tmp[2]), float(error_tmp[3])
        pt1_corrected = pt1 + np.array([[error_x1],[error_y1]])
        pt2_corrected = pt2 + np.array([[error_x2],[error_y2]])
        
        # pixel distance between scene point in image 1(sampson corrected) and its corresponding ground truth pixel coord 
        epslon1 = np.array([[error_x1],[error_y1]])
        
        # pixel distance between predicted image 2 point (by scene point in image 1 and H) and its corresponding ground truth pixel coord 
        pt2_predicted = np.dot(H, self.Homogenize(pt1_corrected))
        epslon2 = self.Dehomogenize(pt2_predicted) - pt2
        
        #covariance propagtion for data normalize 
        if normalize:
            covar1 = float(T1[0][0]**2) * np.eye(2)
            covar2 = float(T2[0][0]**2) * np.eye(2)
        else:
            covar1 = np.eye(2)
            covar2 = np.eye(2)
       
        error = np.transpose(epslon1)@np.linalg.inv(covar1)@epslon1 + np.transpose(epslon2)@np.linalg.inv(covar2)@epslon2
    
        return error
    
    def computeCost(self, H, pts1, pts2, T1, T2, normalize=True):
        # Inputs:
        #    H - camera projection matrix
        #    pts1 - 2D inhomogeneous image1 points
        #    pts2 - 2D inhomogeneous image2 points
        #    tol - Sampson error tolerance 
        #
        # Output:
        #    cost - total Sampson error
         
        cost = 0
        numPoint = pts1.shape[1] 
        for i in range(numPoint):
            #compute error
            if normalize:
                error = self.computeError(pts1[:,i:i+1], pts2[:,i:i+1], H, T1, T2, normalize=True)
            else:
                error = self.computeError(pts1[:,i:i+1], pts2[:,i:i+1], H, T1, T2, normalize=False)
            cost += error
        return cost
    
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
        #    x2 - inhomogeneous inlier correspondences in image 1
        #    normalize - if True, apply data normalization to x1 and x2
        #
        # Outputs:
        #    H - the DLT estimate of the planar projective transformation   
        #    cost - linear estimate cost
        
        """your code here"""
        numPoint = np.shape(x1)[1]
        
        # data normalization
        if normalize:
            print('normalize')
            x1, T1 = self.Normalize(x1) #homo
            x2, T2 = self.Normalize(x2) #homo
        else: 
            T1 = np.eye(3)
            T2 = np.eye(3)
            x1 = self.Homogenize(x1)
            x2 = self.Homogenize(x2)
        
        A = np.zeros((2*numPoint, 9))
        for i in range(numPoint):
            point1 = x1[:, i:i+1]
            point2 = x2[:, i:i+1]
            x_leftNull = self.leftNull(point2)
    
            A[2*i:2*i+2, :] = np.kron(x_leftNull, np.transpose(point1))
       
        u, s, v = np.linalg.svd(A)
        v = v[8:,:]
        
        H = np.reshape(v, (3,3))
            
        # data denormalize
        if normalize:
            print('denormalize')
            #H = np.linalg.inv(T2)@H@T1
            cost = self.computeCost(H, self.Dehomogenize(x1), self.Dehomogenize(x2), T1, T2, normalize=True)
        else: 
            cost = self.computeCost(H, self.Dehomogenize(x1), self.Dehomogenize(x2), T1, T2, normalize=False)
        
        return H, cost


