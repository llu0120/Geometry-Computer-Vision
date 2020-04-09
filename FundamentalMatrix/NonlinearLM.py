from utility import *

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
        return self.DeParameterizeHomog(h).reshape(3,4)
    
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
    
    def getCorrectPoints(self, F, x1, x2): 
        # Input:
        #    F - DLT estimate of the fundamental matrix
        #    x1 - homogeneous inlier points in image 1
        #    x2 - homogeneous inlier points in image 2
        # Output:
        #    x1_hat - corrected points in image 1 
        #    x2_hat - corrected points in image 2
        numPoint = len(x1[0])
        x_hat = np.zeros((3, numPoint))
        x_hat_prime = np.zeros((3, numPoint))
        for i in range(numPoint):
            #Transform that map points to origin 
            x, y, w = float(x1[0, i:i+1]), float(x1[1, i:i+1]), float(x1[2, i:i+1])
            x_prime, y_prime, w_prime = float(x2[0, i:i+1]), float(x2[1, i:i+1]), float(x2[2, i:i+1])
            T = np.array([[w, 0, -x],
                           [0, w, -y],
                           [0, 0,  w]])
            T_prime = np.array([[w_prime, 0       , -x_prime],
                           [0      , w_prime , -y_prime],
                           [0      , 0       ,  w_prime]])
            Fs = np.transpose(np.linalg.inv(T_prime))@F@np.linalg.inv(T)
            
            #Calculate Epipoles e and e' from Fs
            U, S, VT = np.linalg.svd(Fs) # Fe = 0
            e = np.transpose(VT[2:3, :]) #(e1, e2, e3).T
            e1, e2, e3 = float(e[0,0]), float(e[1,0]), float(e[2,0])
            e = np.sqrt(1/(e1**2 + e2**2)) * e
            
            U_prime, S_prime, VT_prime = np.linalg.svd(np.transpose(Fs)) #(F).T e' = 0
            e_prime = np.transpose(VT_prime[2:3, :])                    #(e1', e2', e3').T
            e1_prime, e2_prime, e3_prime = float(e_prime[0,0]), float(e_prime[1,0]), float(e_prime[2,0])
            e_prime = np.sqrt(1/(e1_prime**2 + e2_prime**2)) * e_prime
                
            #Form Rotation Matrices 
            R = np.array([[e1 , e2, 0],
                          [-e2, e1, 0],
                          [  0,  0, 1]])
            R_prime = np.array([[e1_prime , e2_prime, 0],
                                [-e2_prime, e1_prime, 0],
                                [        0,        0, 1]])
            Fs = R_prime@Fs@np.transpose(R)
            
            
            #Form Polynomial g(t) of degree 6
            a, b, c, d, f, f_prime = Fs[1, 1], Fs[1, 2], Fs[2, 1], Fs[2, 2], e3, e3_prime
            
            t = Symbol('t') 
            answer = list(solveset(t*((a*t+b)**2 + f_prime**2*(c*t+d)**2)**2 - (a*d-b*c)*(1+f**2*t**2)**2*(a*t+b)*(c*t+d),t))
            answer = np.array(answer).astype(np.complex64)
            answer = np.real(answer)
    
            #Evaluate Cost Function 
            cost = np.inf
            t_best = 0 
            for j in range(len(answer)+1):
                if j == len(answer):   # t = infinity
                    t = np.inf
                    St = (1/f**2) + c**2/(a**2 + f_prime**2 + c**2)
                else:
                    t = answer[j]
                    St = (t**2/(1+f**2*t**2)) + ((c*t+d)**2/((a*t+b)**2 + f_prime**2*(c*t+d)**2))
                if St < cost: 
                    cost = St 
                    t_best = t
            
            if t_best == np.inf: 
                l = np.array([[f], [0], [-1]])
                l_prime = np.array([[-f_prime*c], [a], [c]])
            else:
                l = np.array([[t_best*f], [1], [-t_best]])
                l_prime = np.array([[-f_prime*(c*t_best+d)], [a*t_best+b], [c*t_best+d]])
    
            #Determine x_hat and x_hat' as the closest points on the lines l and l' to the origin
            a, b, c = l[0,0], l[1,0], l[2,0]
            a_prime, b_prime, c_prime = l_prime[0,0], l_prime[1,0], l_prime[2,0]
            x_pt = np.array([[-a*c], [-b*c], [a**2+b**2]])
            x_prime_pt = np.array([[-a_prime*c_prime], [-b_prime*c_prime], [a_prime**2+b_prime**2]])
            
            #Corrected Points mapped back to Original Coordinates 
            x_hat[:, i:i+1] = np.linalg.inv(T)@np.transpose(R)@x_pt
            x_hat_prime[:, i:i+1] = np.linalg.inv(T_prime)@np.transpose(R_prime)@x_prime_pt
            
        return x_hat, x_hat_prime 
    
    def computeP_prime(self, F): 
        #compute camera projection matrix for image 2 
        U, D, VT = np.linalg.svd(F) 
        s, t = float(D[0]), float(D[1])
        D_prime = np.array([[s, 0,       0],
                            [0, t,       0],
                            [0, 0, (s+t)/2]])
        W = np.array([[ 0, 1, 0],
                      [-1, 0, 0],
                      [ 0, 0, 0]])
        Z = np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]])
        S = U@W@np.transpose(U)
        m = U@Z@D_prime@VT
        e_prime = np.array([[S[2,1]], [S[0,2]], [S[1,0]]])
        P_prime = np.hstack((m, e_prime))
        
        return P_prime 
    
    def TwoViewTriangulation(self, F, P_prime, x_hat, x_hat_prime):
        #Calculate 3D Scene Point
        # Input:
        #    F - DLT estimate of the fundamental matrix
        #    P_prime - camera projection matrix for image 2 
        #    x_hat - corrected homogeneous inlier points in image 1
        #    x_hat_prime - corrected homogeneous inlier points in image 2
        # Output:
        #    x_scene - 3D scene points
        numPoint = len(x_hat[0])
        x_scene = np.zeros((4,numPoint))
        for i in range(numPoint): 
            #epipolar line in image 2
            l_prime = F@x_hat[:, i:i+1] 
            x, y, w = float(x_hat[0, i:i+1]), float(x_hat[1, i:i+1]), float(x_hat[2, i:i+1])
            x_prime, y_prime, w_prime = float(x_hat_prime[0, i:i+1]), float(x_hat_prime[1, i:i+1]), float(x_hat_prime[2, i:i+1])
            a_prime, b_prime, c_prime = float(l_prime[0,0]), float(l_prime[1,0]), float(l_prime[2,0])
            
            #the line perpendicular to the epipolar line in image 2
            l_prime_perpendicular = np.array([[-b_prime*w_prime], 
                                              [a_prime*w_prime],
                                              [b_prime*x_prime-a_prime*y_prime]])   
            
            #back project l_prime_perpendicular to a plane pi
            pi = np.transpose(P_prime)@l_prime_perpendicular #(a, b, c, d).T
            
            #Intersection of 3D line and 3D plane 
            a, b, c, d = float(pi[0,0]), float(pi[1,0]), float(pi[2,0]), float(pi[3,0])
            x_pi = np.array([[d*x], [d*y], [d*w], [-(a*x+b*y+c*w)]])
            x_scene[:, i:i+1] = x_pi
            
        return x_scene
    
    def computeError(self, epsilon, epsilon_prime, covar, covar_prime):
        error = (np.transpose(epsilon)@np.linalg.inv(covar)@epsilon 
                + np.transpose(epsilon_prime)@np.linalg.inv(covar_prime)@epsilon_prime)
        return error 
    
    def computeCost(self, epsilon, epsilon_prime, covar, covar_prime, numPoint): 
        cost = 0
        for i in range(numPoint):
            cost += self.computeError(epsilon[:,i:i+1], epsilon_prime[:,i:i+1], covar, covar_prime)
        return cost 
    
    def skewsymmetric(self, x):
        #Input: 
        #    x - 3D vector (x1, x2, x3).T
        #Output: 
        #    skew-symmetric matrix of x 
        x1, x2, x3 = x[0,0], x[1,0], x[2,0]
        skewsymmetric = np.array([[  0, -x3,  x2],
                                  [ x3,   0, -x1],
                                  [-x2,  x1,   0]])
        return skewsymmetric
    
    def computeA(self, x_scene, P_prime):
        # Input:
        #    x_scene_parameterized -  a parameterized scene point 3 x 1
        #    P_prime - camera projection matrix in image 2
        # Output:
        #    Ai block - 2 x 11
    
        zeroT = np.zeros((1,4))
        x_sceneT = np.transpose(x_scene) # 1 x 4
        
        x2 = P_prime@x_scene                         # 3 x 1
        dehomo_x2 = self.Dehomogenize(x2)                 # 2 x 1
        
        #partial derivative of x(inhomog) to Parameterized P' --> 2 x 12
        w_prime = float(np.dot(P_prime[2:3,:], x_scene))
        dxp1_tmp = np.hstack((x_sceneT, zeroT))
        dxp1 = np.hstack((dxp1_tmp, -float(dehomo_x2[0])*x_sceneT))
        dxp2_tmp = np.hstack((zeroT, x_sceneT))
        dxp2 = np.hstack((dxp2_tmp, -float(dehomo_x2[1])*x_sceneT))
        dxp = (1/w_prime) * np.vstack((dxp1, dxp2))
        
        #Deparameterization of homogenous vector 
        #partial derivative of Parameterized H to Parameter vector h --> 12 x 11 
        v_bar = np.reshape(P_prime, (-1, 1), order='C') 
        v = self.Parameterize(P_prime)
        v_norm = np.linalg.norm(v)
        vT = np.transpose(v)
        
        a = v_bar[0, 0]
        b = v_bar[1:, :]
        
        bLen = len(b)
        if np.linalg.norm(v) == 0: 
            dav = np.zeros((1, bLen)) # 1 x 11
            dbv = 0.5 * np.eye(bLen)  # 11 x 11
        else: 
            dav = -0.5 * np.transpose(b)
            dbv = (self.Sinc(v_norm/2)/2)*np.eye(bLen) + (1/(4*v_norm))*self.dSincdx(v_norm/2)*np.dot(v, vT)
    
        v_barv = np.vstack((dav, dbv)) # 12 x 11
        Ai = np.dot(dxp, v_barv)
    
        return Ai
    
    def computeB(self, x_scene, P):
        # Input:
        #    x_scene - a homogeneous 3D scene point 4 x 1
        #    P - camera projection matrix of image 1 or 2
        # Output:
        #    Bi block - 2 x 2
        #project from scene point to 2D image point 
        x_proj = P@x_scene
        
        #partial derivative of x(inhomog) to Parameterized xs(scene point) --> 2 x 4
        p1T = P[0:1,:]
        p2T = P[1:2,:]
        p3T = P[2:3,:]
        w_prime = float(np.dot(p3T, x_scene))
    
        dxpx1 = p1T - float(self.Dehomogenize(x_proj)[0])*p3T
        dxpx2 = p2T - float(self.Dehomogenize(x_proj)[1])*p3T
        dxpx = (1/w_prime) * np.vstack((dxpx1, dxpx2))
    
        #Deparameterization of homogenous vector
        #partial derivative of Parameterized xs to Parameter vector xs --> 4 x 3
        v_bar = np.reshape(x_scene, (-1, 1), order='C') 
        v = self.Parameterize(x_scene)
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
    
    def scenePtInitialization(self, F, x1, x2): 
        numPoint = len(x1[0])
        x1 = self.Homogenize(x1) 
        x2 = self.Homogenize(x2)
        
        # Initialize 3D Scene points using 2-view "optimal" triangulation
        # camera projection matrix in image 2
        P_prime = self.computeP_prime(F)
        
        # corrected points x1, x2 
        print("Start correcting x1, x2...")
        x_hat, x_hat_prime = self.getCorrectPoints(F, x1, x2)
        print("Finish correcting.")
        
        # scene points 
        x_scene = self.TwoViewTriangulation(F, P_prime, x_hat, x_hat_prime)
        print("Finish intializing the 3D scene points.")
        
        return x_scene, P_prime
    
    def LM(self, F, x1, x2, x_scene, P_prime, max_iters, lam):
        # Input:
        #    F - DLT estimate of the fundamental matrix
        #    x1 - inhomogeneous inlier points in image 1
        #    x2 - inhomogeneous inlier points in image 2
        #    x_scene - inhomogeneous scene points
        #    max_iters - maximum number of iterations
        #    lam - lambda parameter
        # Output:
        #    F - Final fundamental matrix obtained after convergence
        """your code here"""
        numPoint = len(x1[0])
        # camera projection matrix in image 1 
        P = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
        
        # Data Normalization 
        x1, T1 = self.Normalize(x1)
        x2, T2 = self.Normalize(x2)
        x_scene, Ts = self.Normalize(self.Dehomogenize(x_scene))
        
        s1 = T1[0][0]
        s2 = T2[0][0]
        s3 = Ts[0][0]
        
        covar = (s1**2) * np.eye(2)
        covar_prime = (s2**2) * np.eye(2)
        
        # data normalization of camera projection matrix in image 1 and 2 
        P = T1@P@np.linalg.inv(Ts)
        P_prime = T2@P_prime@np.linalg.inv(Ts)
        
        #Parameterize P, P' and scene points 
        p = self.Parameterize(P)
        p_prime = self.Parameterize(P_prime)
        x_scene_parameterized = np.zeros((np.shape(x_scene)[0]-1, np.shape(x_scene)[1])) 
        
        for i in range(numPoint):
            x_scene_parameterized[:, i:i+1] = self.Parameterize(x_scene[:,i:i+1]) 
            
        #initialize of the cost
        epsilon = self.Dehomogenize(x1) - self.Dehomogenize(P@x_scene)
        epsilon_prime = self.Dehomogenize(x2) - self.Dehomogenize(P_prime@x_scene)
        cost_prev = self.computeCost(epsilon, epsilon_prime, covar, covar_prime, numPoint)
        print("Cost Initialized.")
        
        print ('iter %03d Cost %.9f'%(0, cost_prev))
        
        i = 0  
        
        while i < max_iters: 
            U_prime = np.zeros((11,11))
            epsilon_a = np.zeros((11,1))
            WV_starInvWT = np.zeros((11,11))
            WV_starInvEpsb = np.zeros((11,1))
            V = np.zeros((3*numPoint,3))
            
            W_prime = np.zeros((11*numPoint,3))
            
            epsilon_b = np.zeros((3*numPoint,1))
            
            for n in range(numPoint):
                #Construct Ai, Bi, Bi_prime for jacobian matrix (Sparse)
                Ai_prime = self.computeA(x_scene[:,n:n+1], P_prime)    # 2 x 11
                
                Bi = self.computeB(x_scene[:,n:n+1], P)          # 2 x 3 
                Bi_prime = self.computeB(x_scene[:,n:n+1], P_prime)    # 2 x 3
                
                #Construct Normal Matrix
                U_prime += np.transpose(Ai_prime) @ np.linalg.inv(covar_prime) @ Ai_prime  # 11 x 11
                Vi = (np.transpose(Bi) @ np.linalg.inv(covar) @ Bi                         # 3 x 3 
                    + np.transpose(Bi_prime) @ np.linalg.inv(covar_prime) @ Bi_prime)
                V[3*n:3*n+3,:] = Vi
                
                Wi_prime = np.transpose(Ai_prime) @ np.linalg.inv(covar_prime) @ Bi_prime  # 11 x 3 
                W_prime[11*n:11*n+11,:] = Wi_prime
                
                #Construct Normal Vector 
                epsilon_i = self.Dehomogenize(x1[:,n:n+1]) - self.Dehomogenize(P@x_scene[:,n:n+1])
                epsilon_i_prime = self.Dehomogenize(x2[:,n:n+1]) - self.Dehomogenize(P_prime@x_scene[:,n:n+1])
                
                epsilon_a += np.transpose(Ai_prime) @ np.linalg.inv(covar_prime) @ epsilon_i_prime # 11 x 1
                epsilon_bi = (np.transpose(Bi) @ np.linalg.inv(covar) @ epsilon_i                  
                             + np.transpose(Bi_prime) @ np.linalg.inv(covar_prime) @ epsilon_i_prime) # 2 x 1
                epsilon_b[3*n:3*n+3,:]  = epsilon_bi
                
                WV_starInvWT += Wi_prime @ np.linalg.inv(Vi + lam * np.eye(3)) @ np.transpose(Wi_prime) # 11 x 11
                WV_starInvEpsb += Wi_prime @ np.linalg.inv(Vi + lam * np.eye(3)) @ epsilon_bi # 11 x 1
                
            #Augmented Normal Equations 
            S_prime = (U_prime + lam * np.eye(11)) - WV_starInvWT                             # 11 x 11
            e_prime = epsilon_a - WV_starInvEpsb                                             # 11 x 1
                
            #Adjust p'
            delta_a_prime = np.linalg.inv(S_prime) @ e_prime                                 # 11 x 1
            p_prime0 = p_prime + delta_a_prime 
            
            #Adjust parameterized scene point 
            x_scene0_parmaterized = np.zeros((3, numPoint))
            for n in range(numPoint):
                delta_bi = np.linalg.inv(V[3*n:3*n+3,:] + lam * np.eye(3)) @ (epsilon_b[3*n:3*n+3,:] 
                                                             - np.transpose(W_prime[11*n:11*n+11,:]) @ delta_a_prime)
                x_scene0_parmaterized[:, n:n+1] = x_scene_parameterized[:,n:n+1] + delta_bi
            
            #Deparameterize p'
            P_prime0 = self.Deparameterize(p_prime0)
            
            #Deparameterize scene point
            x_scene0 = np.zeros((4,numPoint))
            for n in range(numPoint):
                x_scene0[:, n:n+1] = self.DeParameterizeHomog(x_scene0_parmaterized[:,n:n+1]) 
                
            epsilon = self.Dehomogenize(x1) - self.Dehomogenize(P@x_scene0)
            epsilon_prime = self.Dehomogenize(x2) - self.Dehomogenize(P_prime0@x_scene0)
            cost0 = self.computeCost(epsilon, epsilon_prime, covar, covar_prime, numPoint)
    
            #Adjust lambda 
            if cost0 > cost_prev: 
                lam = 10 * lam 
            else: # cost0 < cost_prev: 
                if abs(cost_prev - cost0) < 10**(-12):
                    print("The cost converged.")
                    break
                lam = 0.1 * lam 
                p_prime = p_prime0 
                P_prime = P_prime0
                x_scene = x_scene0 
                x_scene_parameterized = x_scene0_parmaterized
                cost_prev = cost0
                print ('iter %03d Cost %.9f'%(i+1, cost_prev))
                i += 1 
            
        
        P_prime = self.Deparameterize(p_prime)
        #Data Denormalize P_prime 
        P_prime = np.linalg.inv(T2) @ P_prime @ Ts
        
        # From P' calcuate F 
        e_prime = P_prime[:,3:4]
        e_prime_skew = self.skewsymmetric(e_prime)
        m = P_prime[:,0:3]
        
        F = e_prime_skew @ m 
        
        return F
    
    