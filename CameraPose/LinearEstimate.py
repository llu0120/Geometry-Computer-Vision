from utility import *

'''
Linear Estimation of camera pose by EPnP Algorithm
'''
class LinearEstimate(): 
    
    def Homogenize(self, x):
        # converts points from inhomogeneous to homogeneous coordinates
        return np.vstack((x,np.ones((1,x.shape[1]))))

    def Dehomogenize(self, x):
        # converts points from homogeneous to inhomogeneous coordinates
        return x[:-1]/x[-1] 
    
    def ComputeCost(self, P, x, X, K):
        # Inputs:
        #    P - camera projection matrix
        #    x - 2D groundtruth image points
        #    X - 3D groundtruth scene points
        #    K - camera calibration matrix
        #
        # Output:
        #    cost - total projection error
        n = x.shape[1]
        covarx = np.eye(2*n) # covariance propagation
        
        """your code here"""
        x_hat = K@P@self.Homogenize(X)
        
     
        epslon = x - self.Dehomogenize(x_hat)
        epslon = np.reshape(epslon, (2*n, 1), 'F') #2n x 1
        epslonT = np.transpose(epslon) #1 x 2n
        
        cost = np.dot(np.dot(epslonT, np.linalg.inv(covarx)), epslon)
        
        return float(cost)
    
    def EPnP(self, x, X, K):
        # Inputs:
        #    x - 2D inlier points
        #    X - 3D inlier points
        # Output:
        #    P - normalized camera projection matrix
        
        """your code here"""
        numPoint = X.shape[1]
    
        #Compute mean and cov for X
        X_mean = np.transpose(np.array([np.mean(X, axis = 1)]))
        X_cov = np.cov(X)
        U, S, Vh = np.linalg.svd(X_cov)
        V = np.transpose(Vh)
        v1 = V[:, 0:1]
        v2 = V[:, 1:2]
        v3 = V[:, 2:3]
        
        #Compute total variance 
        total_var = np.sum(S) #Trace of S
        
        #Four control points 
        C1 = X_mean
        C2 = v1 + X_mean
        C3 = v2 + X_mean
        C4 = v3 + X_mean
        m = np.zeros((2 * numPoint, 12))
        alpha = np.zeros((4, numPoint))
    
        #A = np.hstack((np.hstack((C2 - C1, C3 - C1)), C4 - C1))
        for i in range(numPoint):
            #Parameterization of 3D Points: alpha1, alpha2, alpha3, alpha4
            b = X[:, i:i+1] - C1 
            alpha_vec = np.dot(Vh, b)
            
            alpha2 = float(alpha_vec[0])
            alpha3 = float(alpha_vec[1])
            alpha4 = float(alpha_vec[2])
            alpha1 = float(1 - alpha2 - alpha3 - alpha4)
            alpha[1:4, i:i+1] = alpha_vec
            alpha[0:1, i:i+1] = alpha1
            
            #Control points in camera coordinate
            x_p = np.dot(np.linalg.inv(K), self.Homogenize(x[:, i:i+1]))
            x_p = self.Dehomogenize(x_p)
    
            xi = float(x_p[0])
            yi = float(x_p[1])
            m_p = np.array([[alpha1, 0, -alpha1*xi, alpha2, 0, -alpha2*xi, alpha3, 0, -alpha3*xi, alpha4, 0, -alpha4*xi],
                            [0, alpha1, -alpha1*yi, 0, alpha2, -alpha2*yi, 0, alpha3, -alpha3*yi, 0, alpha4, -alpha4*yi]])
            m[2*i:2*i+2, :] = m_p
    
        U, S, Vh = np.linalg.svd(m)
        V = np.transpose(Vh)
    
        C_cam1 = V[0:3, 11:12]
        C_cam2 = V[3:6, 11:12]
        C_cam3 = V[6:9, 11:12]
        C_cam4 = V[9:12, 11:12]
    
        #Deparameterize 3D points in camera coordinate
        X_cam = np.zeros((3, numPoint))
        for i in range(numPoint):
            X_cam[:, i:i+1] = alpha[0, i] * C_cam1 + alpha[1, i] * C_cam2 + alpha[2, i] * C_cam3 + alpha[3, i] * C_cam4
    
        #Fix the scale probelm 
        X_cam_mean = np.transpose(np.array([np.mean(X_cam, axis = 1)]))
        X_cam_cov = np.cov(X_cam)
    
        U, S, Vh = np.linalg.svd(X_cam_cov)
        total_var_cam = np.sum(S) #Trace of S
        
        if float(X_cam_mean[2]) < 0:
            beta = -np.sqrt(total_var / total_var_cam)
        else:
            beta = np.sqrt(total_var / total_var_cam)
    
        X_cam = beta * X_cam
        
        
        ######### Iterative Closest Point ###########
        X_cam_mean = np.transpose(np.array([np.mean(X_cam, axis = 1)]))    
        
        q = X - X_mean
        q_cam = X_cam - X_cam_mean
        
        w = np.zeros((3, 3))
        for i in range(numPoint):
            w += np.dot(q_cam[:, i:i+1], np.transpose(q[:, i:i+1]))
        
        U, _, Vh = np.linalg.svd(w)
        V = np.transpose(Vh)
        if (np.linalg.det(U) * np.linalg.det(V) < 0):
            S = np.eye(3)
            S[2,2] = -1
            R = np.dot(np.dot(U, S), Vh)
        else:
            R = np.dot(U, Vh)
        t = X_cam_mean - np.dot(R, X_mean)        
        P = np.concatenate((R, t), axis=1)
    
        
        return P


