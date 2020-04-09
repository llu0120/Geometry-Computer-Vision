from utility import *
'''
Nonlinear Optimization of camera pose by Lavenberg-Marquardt (LM)
'''
class NonlinearLM(): 
    
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
    
    # Note that np.sinc is different than defined in class
    def Sinc(self, x):
        """your code here"""
        if (x == 0):
            y = 1
        else:
            y = np.sin(x)/x
        
        return y
    
    def dSincdx(self, x): 
        if x == 0: 
            y = 0
        else: 
            y = (np.cos(x)/x) - (np.sin(x)/(x*x))
        
        return y
    
    def dsdtheta(self, x):
        y = (x * np.sin(x) - 2 * (1 - np.cos(x))) / (x**3)
        
        return y 
    
    def dthetadomega(self, w):
        theta = np.linalg.norm(w)
        y = (1/theta) * np.transpose(w)
        
        return y
    
    def skew(self, w):
        # Returns the skew-symmetrix represenation of a vector
        """your code here"""
        w1, w2, w3 = float(w[0]), float(w[1]), float(w[2])
        w_skew = np.zeros((3, 3))
        w_skew[0][1] = -w3
        w_skew[0][2] = w2
        w_skew[1][0] = w3
        w_skew[1][2] = -w1
        w_skew[2][0] = -w2
        w_skew[2][1] = w1 
        
        return w_skew
    
    
    def Parameterize(self, R):
        # Parameterizes rotation matrix into its axis-angle representation
        """your code here"""
        w_skew = logm(R)
    
        w1 = float(-w_skew[1][2])
        w2 = float(w_skew[0][2])
        w3 = float(w_skew[1][0])
        w = np.array([[w1],
                      [w2],
                      [w3]])
        theta = np.linalg.norm(w)
    
        return w, theta
    
    
    def Deparameterize(self, w):
        # Deparameterizes to get rotation matrix
        """your code here"""
        theta = np.linalg.norm(w)
        R = np.cos(theta) * np.eye(3) +self.Sinc(theta) * self.skew(w) + ((1 - np.cos(theta)) / (theta**2)) * w * np.transpose(w)
        
        return R
    
    def cross(self, w, X_point):
        '''
        Input: 
        w       3x1 vec
        X_point 3x1 vec 
        
        Output: 
        cross product 3x1 
        '''
        w = np.transpose(w)
        X_point = np.transpose(X_point)
        tmp = np.cross(w, X_point)
    
        return np.transpose(tmp)

    def Jacobian(self, R, w, t, X):
        # compute the jacobian matrix
        # Inputs:
        #    R - 3x3 rotation matrix
        #    w - 3x1 axis-angle parameterization of R
        #    t - 3x1 translation vector
        #    X - 3D inlier points
        #
        # Output:
        #    J - Jacobian matrix of size 2*nx6
        
        """your code here"""
        angle_thres = 10**(-5)
        numPoint = X.shape[1]
        J = np.zeros((2*numPoint,6))
        for i in range(numPoint):
            #Project inhomogeneous 3D point into homogeneous 2D point in normalized coordinates
            X_point = X[: ,i:i+1]
            x_normalize_inhomog = self.Dehomogenize(np.dot(R, X_point) + t)
            
            theta = np.linalg.norm(w)
    
            #Compute the part of partial x to omega
            t3 = float(t[2])
            
            if theta == 0 or theta < angle_thres: 
                X_rotated = X_point + self.cross(w, X_point)
                xrotated_omega = self.skew(-X_point)
                
            else:    
                X_rotated = X_point + self.Sinc(theta) * self.cross(w, X_point) \
                            + ((1 - np.cos(theta))/(theta**2)) * self.cross(w, self.cross(w, X_point))
                
                s = (1 - np.cos(theta)) / (theta**2)
                
                xrotated_omega = self.Sinc(theta) * self.skew(-X_point) + self.cross(w, X_point) * self.dSincdx(theta) * self.dthetadomega(w) \
                                 + self.cross(w, self.cross(w, X_point)) * self.dsdtheta(theta) * self.dthetadomega(w) \
                                 + s * (np.dot(self.skew(w), self.skew(-X_point)) + self.skew(-self.cross(w, X_point)))
                
            omega_head = float(X_rotated[2]) + t3
            x_xrotated = np.array([[1/omega_head, 0, -float(x_normalize_inhomog[0])/omega_head],
                                   [0, 1/omega_head, -float(x_normalize_inhomog[1])/omega_head]])
    
            x_omega = np.dot(x_xrotated, xrotated_omega)
            
            #Compute the part of partial x to t 
            r3T = R[2:3, :]
            omega_head = float(np.dot(r3T, X_point) + t3)
            x_t = np.array([[1/omega_head, 0, -float(x_normalize_inhomog[0])/omega_head],
                            [0, 1/omega_head, -float(x_normalize_inhomog[1])/omega_head]])
            
            J_part = np.concatenate((x_omega, x_t), axis=1)
            J[2*i:2*i+2, :] = J_part
        
        return J
    
    def LM(self, P, x, X, K, max_iters, lam):
        # Inputs:
        #    P - initial estimate of camera pose
        #    x - 2D inliers
        #    X - 3D inliers
        #    K - camera calibration matrix 
        #    max_iters - maximum number of iterations
        #    lam - lambda parameter
        #
        # Output:
        #    P - Final camera pose obtained after convergence
    
        """your code here"""
        numPoint = len(x[0])
        
        #Normalized coordinate
        x_hat = K@P@self.Homogenize(X)
        error_prev = x - self.Dehomogenize(x_hat)
        error_prev = np.reshape(error_prev, (2*numPoint, 1), 'F')
        
        cost_prev = self.ComputeCost(P, x, X, K) #initialize of the cost 
        print ('iter %03d Cost %.20f'%(0, cost_prev))
    
        # you may modify this so long as the cost is computed
        # at each iteration
        i = 0
        while(i < max_iters): 
            #Get Jacobian Matrix
            R = P[0:4, 0:3]
            t = P[0:4, 3:4]
            w, theta = self.Parameterize(R)
            J = self.Jacobian(R, w, t, X)
            #print(J)
            p = np.array([[float(w[0])],
                          [float(w[1])],
                          [float(w[2])],
                          [float(t[0])],
                          [float(t[1])],
                          [float(t[2])]])
            '''
            #Covariance Propagation 
            covariance = np.eye(2*numPoint)
            JK = K[0:2,0:2]
            JKT = np.transpose(JK)
            for i in range(numPoint):
                covar_part = np.dot(np.dot(JK, covariance[2*i:2*i+2, 2*i:2*i+2]), JKT)
                covariance[2*i:2*i+2, 2*i:2*i+2] = covar_part
            '''
            covariance = np.eye(2*numPoint)
            #Form normal equation 
            norm_Mat = np.dot(np.dot(np.transpose(J), np.linalg.inv(covariance)), J) + lam*np.eye(6)
            norm_vec = np.dot(np.dot(np.transpose(J), np.linalg.inv(covariance)), error_prev)
            delta = np.dot(np.linalg.inv(norm_Mat), norm_vec)
            p0 = p + delta
            w0 = np.array([[float(p0[0])],
                           [float(p0[1])],
                           [float(p0[2])]])
            R0 = self.Deparameterize(w0)
            t0 = np.array([[float(p0[3])],
                           [float(p0[4])],
                           [float(p0[5])]])
            
            P0 = np.concatenate((R0, t0), axis=1)
    
            cost0 = self.ComputeCost(P0, x, X, K)
            if cost0 > cost_prev: 
                lam = 10*lam
                #Optimization stop 
                if lam > 10**20: 
                    P = P0 
                    i += 1
                    for i in range(i, max_iters+1):
                        print('iter %03d Cost %.20f'%(i, cost_prev))
                        i += 1
            else: 
                lam = 0.1*lam
                i += 1
                p = p0
                P = P0
                x_hat = np.dot(P, self.Homogenize(X))
                error_prev = x - self.Dehomogenize(x_hat)
                error_prev = np.reshape(error_prev, (2*numPoint, 1), 'F')
    
                cost_prev = cost0
    
                print ('iter %03d Cost %.20f'%(i, cost_prev))
                
        #Denormalize P matrix 
        P = np.dot(K, P)
        return P


      
      