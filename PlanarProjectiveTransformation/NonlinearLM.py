from utility import *
"""
Nonlinear Optimization of Lavenberg-Marquar
"""
class NonlinearLM(): 
    
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
    
    # Note that np.sinc is different than defined in class
    def Sinc(self, x):
        # Returns a scalar valued sinc value
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
    
    def Parameterize(self, H):
        # wrapper function to interface with LM
        # takes all optimization variables and parameterizes all of them
        # in this case it is just P, but in future assignments it will
        # be more useful
        return self.ParameterizeHomog(H.reshape(-1,1))
    
    
    def Deparameterize(self, h):
        # Deparameterize all optimization variables
        return self.DeParameterizeHomog(h).reshape(3,3)
    
    
    def ParameterizeHomog(self, V):
        # Given a homogeneous vector V return its minimal parameterization
        """your code here"""
        V = V / np.linalg.norm(V)
        a = V[0][0]
        b = V[1:len(V), :]
        v = (2/self.Sinc(np.arccos(a))) * b
        v_norm = np.linalg.norm(v)
        if v_norm > np.pi: 
            v = (1 - (2*np.pi/v_norm)*np.ceil((((v_norm - np.pi)/(2*np.pi)))))* v
        
        return v
    
    
    def DeParameterizeHomog(self, v):
        # Given a parameterized homogeneous vector return its deparameterization
        """your code here"""
        v_norm = np.linalg.norm(v)
        a = np.cos(v_norm/2)
        b = (self.Sinc(v_norm/2) / 2) * np.transpose(v)
        v_bar = np.zeros((len(v)+1,1))
        
        v_bar[0,0] = a
        v_bar[1:, :] = np.transpose(b)
        v_bar = v_bar / np.linalg.norm(v_bar)
        
        return v_bar
    
    def sampsonCorrection(self, pt1, pt2, H): #Initialize 2D scene points by sampson correction
        # Inputs:
        #    pt1 - a 2D inhomogeneous image1 point
        #    pt2 - a 2D inhomogeneous image2 point
        #
        # Output:
        #    pt1_corrected
        #    pt2_corrected 
        
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
        
        lamda = np.dot(np.linalg.inv(np.dot(J, JT)), (-epslon))
        error_tmp = np.dot(JT, lamda)
        
        #Sampson correction 
        error_x1, error_y1 = float(error_tmp[0]), float(error_tmp[1])
        error_x2, error_y2 = float(error_tmp[2]), float(error_tmp[3])
        pt1_corrected = pt1 + np.array([[error_x1],[error_y1]])
        pt2_corrected = pt2 + np.array([[error_x2],[error_y2]])
        
        return pt1_corrected, pt2_corrected
    
    
    def computeA(self, x_scene_pt, H):
        # Input:
        #    x_scene_pt 
        #    H - DLT estimate of planar projective transformation matrix
        # Output:
        #    Ai block - 2 x 8
    
        zeroT = np.zeros((1,3))
        x = self.DeParameterizeHomog(x_scene_pt)
        #x = Homogenize(x_scene_pt)
    
        xT = np.transpose(x)
        
        x2 = H@x
        dehomo_x2 = self.Dehomogenize(x2)
        
        v_bar = np.reshape(H, (-1, 1), order='C')
        v = self.Parameterize(H)
        v_norm = np.linalg.norm(v)
        vT = np.transpose(v)
    
        #partial derivative of x(inhomog) to Parameterized H --> 2 x 9
        w_prime = float(np.dot(H[2:3,:], x))
        dxph1_tmp = np.hstack((xT, zeroT))
        dxph1 = np.hstack((dxph1_tmp, -float(dehomo_x2[0])*xT))
        dxph2_tmp = np.hstack((zeroT, xT))
        dxph2 = np.hstack((dxph2_tmp, -float(dehomo_x2[1])*xT))
        dxph = (1/w_prime) * np.vstack((dxph1, dxph2))
        
        #Deparameterization of homogenous vector 
        #partial derivative of Parameterized H to Parameter vector h --> 9 x 8 
        a = v_bar[0, 0]
        b = v_bar[1:, :]
        
        bLen = len(b)
        if np.linalg.norm(v) == 0: 
            dav = np.zeros((1, bLen)) # 1 x 8
            dbv = 0.5 * np.eye(bLen)  # 8 x 8
        else: 
            dav = -0.5 * np.transpose(b)
            dbv = (self.Sinc(v_norm/2)/2)*np.eye(bLen) + (1/(4*v_norm))*self.dSincdx(v_norm/2)*np.dot(v, vT)
    
        v_barv = np.vstack((dav, dbv)) # 2 x 8
        Ai = np.dot(dxph, v_barv)
    
        return Ai
    
    
    
    def computeB(self, x_scene_pt, H):
        # Input:
        #    x_scene_pt - a inhomogeneous inlier point in image 1 or 2(this is sampson corrected)
        #    H - DLT estimate of planar projective transformation matrix
        # Output:
        #    Bi block - 2 x 2
        
        #partial derivative of x(inhomog) to Parameterized xs(scene point) --> 2 x 3
        h1T = H[0:1,:]
        h2T = H[1:2,:]
        h3T = H[2:3,:]
        w_prime = float(np.dot(h3T, self.Homogenize(x_scene_pt)))
        #w_prime = float(np.dot(h3T, DeParameterizeHomog(x_scene_pt)))
    
        dxpx1 = h1T - float(x_scene_pt[0])*h3T
        dxpx2 = h2T - float(x_scene_pt[1])*h3T
        dxpx = (1/w_prime) * np.vstack((dxpx1, dxpx2))
    
        #Deparameterization of homogenous vector
        #partial derivative of Parameterized xs to Parameter vector xs --> 3 x 2
        v_bar = np.reshape(self.Homogenize(x_scene_pt), (-1, 1), order='C') 
        v = self.Parameterize(self.Homogenize(x_scene_pt))
        vT = np.transpose(v)
        v_norm = np.linalg.norm(v)
        
        a = v_bar[0, 0]
        b = v_bar[1:, :]
    
        bLen = len(b)
        if np.linalg.norm(v) == 0: 
            dav = np.zeros((1, bLen)) # 1 x 2
            dbv = 0.5 * np.eye(bLen)  # 2 x 2
        else: 
            dav = -0.5 * np.transpose(b)
            dbv = (self.Sinc(v_norm/2)/2)*np.eye(bLen) + (1/(4*v_norm))*self.dSincdx(v_norm/2)*np.dot(v, vT)
    
        v_barv = np.vstack((dav, dbv)) # 3 x 2
        Bi = np.dot(dxpx, v_barv)
        
        return Bi
    
    def LM(self, H, x1, x2, max_iters, lam):
        # Input:
        #    H - DLT estimate of planar projective transformation matrix
        #    x1 - inhomogeneous inlier points in image 1
        #    x2 - inhomogeneous inlier points in image 2
        #    max_iters - maximum number of iterations
        #    lam - lambda parameter
        # Output:
        #    H - Final H (3x3) obtained after convergence
        
        # data normalization
        x1, T1 = self.Normalize(x1)
        x2, T2 = self.Normalize(x2)
        #H = T2@H@np.linalg.inv(T1)
        s1 = T1[0][0]
        s2 = T2[0][0]
        """your code here"""
        numPoint = len(x1[0])
        
        #initialize of the cost 
        cost_prev = float(self.computeCost(H, self.Dehomogenize(x1), self.Dehomogenize(x2), T1, T2))
        
        #Initialize Scene points as the sampson corrected data normalized points in the first image 
        x_scene = np.zeros(self.Dehomogenize(x1).shape)
        for n in range(numPoint):
            pt1_corrected, pt2_corrected = self.sampsonCorrection(self.Dehomogenize(x1[:,n:n+1]), self.Dehomogenize(x2[:,n:n+1]), H)
            x_scene[:,n:n+1] = pt1_corrected
        
        i = 0  
        while i < max_iters:        
            h = self.Parameterize(H)
            A = np.zeros((2*numPoint, 8))
            B = np.zeros((2*numPoint, 2, 2))
            B_prime = np.zeros((2*numPoint, 2, 2))
            covar = (s1**2) * np.eye(2)
            covar_prime = (s2**2) * np.eye(2)
            
            U_prime = np.zeros((8,8))
            W_prime = np.zeros((8*numPoint,2))
            V = np.zeros((2*numPoint,2))
            
            epsilon_a = np.zeros((8,1))
            epsilon_b = np.zeros((2*numPoint,1))
        
            WV_starInvWT = np.zeros((8,8))
            WV_starInvEpsb = np.zeros((8,1))
            
            for n in range(numPoint):
                #Construct Ai, Bi, Bi_prime for jacobian matrix (Sparse)
                Ai_prime = self.computeA(x_scene[:,n:n+1], H)             # 2 x 8
                A[2*n:2*n+2,:] = Ai_prime
                
                Bi = self.computeB(x_scene[:,n:n+1], np.eye(3))           # 2 x 2
                B[2*n:2*n+2,:] = Bi
                
                Bi_prime = self.computeB(self.Dehomogenize(H@self.Homogenize(x_scene[:,n:n+1])), H)      # 2 x 2
                B_prime[2*n:2*n+2,:] = Bi_prime
                 
                #Construct Normal Matrix 
                U_prime += np.transpose(Ai_prime) @ np.linalg.inv(covar_prime) @ Ai_prime   # 8 x 8
                Vi = (np.transpose(Bi) @ np.linalg.inv(covar) @ Bi                           
                   + np.transpose(Bi_prime) @ np.linalg.inv(covar_prime) @ Bi_prime)         # 2 x 2
                V[2*n:2*n+2,:] = Vi
                
                Wi_prime = np.transpose(Ai_prime) @ np.linalg.inv(covar_prime) @ Bi_prime   # 8 x 2 
                W_prime[8*n:8*n+8,:] = Wi_prime
                
                #Normal Vector  
                epsilon_i = self.Dehomogenize(x1[:,n:n+1]) - x_scene[:,n:n+1]        # 2 x 1
                epsilon_prime_i = self.Dehomogenize(x2[:,n:n+1]) - self.Dehomogenize(H@self.Homogenize(x_scene[:,n:n+1])) # 2 x 1
                
                epsilon_a += np.transpose(Ai_prime) @ np.linalg.inv(covar_prime) @ epsilon_prime_i # 8 x 1
                epsilon_bi = (np.transpose(Bi) @ np.linalg.inv(covar) @ epsilon_i                  
                           + np.transpose(Bi_prime) @ np.linalg.inv(covar_prime) @ epsilon_prime_i) # 2 x 1
                epsilon_b[2*n:2*n+2,:] = epsilon_bi
                
                WV_starInvWT += Wi_prime @ np.linalg.inv(Vi + lam * np.eye(2)) @ np.transpose(Wi_prime) # 8 x 8
                WV_starInvEpsb += Wi_prime @ np.linalg.inv(Vi + lam * np.eye(2)) @ epsilon_bi # 8 x 1
            
            #Augmented Normal Equations 
            S_prime = (U_prime + lam * np.eye(8)) - WV_starInvWT                             # 8 x 8
            e_prime = epsilon_a - WV_starInvEpsb                                             # 8 x 1
            
            #Adjust h
            delta_a_prime = np.linalg.inv(S_prime) @ e_prime                                 # 8 x 1
            h0 = h + delta_a_prime 
            
            #Adjust scene point
            x_scene0 = x_scene
            for n in range(numPoint):  
                delta_bi = np.linalg.pinv(V[2*n:2*n+2,:]) @ (epsilon_b[2*n:2*n+2,:] 
                                                            - np.transpose(W_prime[8*n:8*n+8,:]) @ delta_a_prime)
                x_scene0[:,n:n+1] += delta_bi 
            
            H0 = self.Deparameterize(h0)
            cost0 = float(self.computeCost(H0, self.Dehomogenize(x1), self.Dehomogenize(x2), T1, T2, normalize=True))
            
            #Adjust lambda 
                
            if cost0 > cost_prev: 
                lam = 10*lam
                if (lam > 10**(30)):
                    print("The cost converged.")
                    break
            else: #cost0 < cost_prev:
                if abs(cost0 - cost_prev) < 10**(-12):
                    print("The cost converged.")
                    break
                lam = 0.1*lam 
                i += 1
                h = h0 
                H = H0
                x_scene = x_scene0
                cost_prev = cost0
                print ('iter %03d Cost %.9f'%(i, cost_prev))
           
        # data denormalization
        H = self.Deparameterize(h)
        H = np.linalg.inv(T2)@H@T1
        return H


