'''
Nonlinear Estimation of the Camera Projection Matrix by Lavenberg-Marquardt (LM) Algorithm
'''
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
    
    def Parameterize(self, P):
        # wrapper function to interface with LM
        # takes all optimization variables and parameterizes all of them
        # in this case it is just P, but in future assignments it will
        # be more useful
        return self.ParameterizeHomog(P.reshape(-1,1))
    
    
    def Deparameterize(self, p):
        # Deparameterize all optimization variables
        return self.DeParameterizeHomog(p).reshape(3,4)
    
    
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
    
    def Jacobian(self, P,p,X):
        # compute the jacobian matrix
        #
        # Input:
        #    P - 3x4 projection matrix
        #    p - 11x1 homogeneous parameterization of P
        #    X - 3n 3D scene points
        # Output:
        #    J - 2nx11 jacobian matrix
        J = np.zeros((2*X.shape[1],11))
        
        """your code here"""
        numPoint = len(X[0])
        X_homo = self.Homogenize(X)
        x_homo = np.dot(P, X_homo)
        dehomo_x = self.Dehomogenize(x_homo)
        
        zeroT = np.zeros((1,4))
        
        v_bar = np.reshape(P, (-1, 1), order='C')
        v = p
        vT = np.transpose(v)
        v_norm = np.linalg.norm(v)
    
        for i in range(numPoint):
            #Projection of a point under the camera projecion matrix 
            #partial derivative of x(inhomog) to Parameterized P --> 2 x 12
            p3T = P[2:3,:]
            w = float(np.dot(p3T, X_homo[:, i:i+1]))
            XT = np.transpose(X_homo[:, i:i+1])
            xp1_tmp = np.hstack((XT, zeroT))
            xp1 = np.hstack((xp1_tmp, -float(dehomo_x[0, i:i+1])*XT))
    
            xp2_tmp = np.hstack((zeroT, XT))
            xp2 = np.hstack((xp2_tmp, -float(dehomo_x[1, i:i+1])*XT))
    
            xp = (1/w) * np.vstack((xp1, xp2)) # 2 x 12
    
            #Deparameterization of homogenous vector
            #partial derivative of Parameterized P to Parameter vector p 
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
            J[2*i:2*i+2, :] = np.dot(xp, v_barv)
        
      
        return J
    
    def ComputeCostNormalized(self, P, x, X, s):
        # Inputs:
        #    x - 2D Data Normalized homogeneous image points
        #    X - 3D Data Normalized homogeneous scene points
        #
        # Output:
        #    cost - Total reprojection error
        n = x.shape[1]
        covarx = (s**2)*np.eye(2*n)
        
        x_hat = np.dot(P, self.Homogenize(X))
        epslon = x - self.Dehomogenize(x_hat)
        
        epslon = np.reshape(epslon, (2*n, 1), 'F') #2n x 1
        epslonT = np.transpose(epslon) #1 x 2n
        cost = np.dot(np.dot(epslonT, np.linalg.inv(covarx)), epslon)
        
        return cost
    
    def LM(self, P, x, X, max_iters, lam):
        # Input:
        #    P - initial estimate of P
        #    x - 2D inhomogeneous image points
        #    X - 3D inhomogeneous scene points
        #    max_iters - maximum number of iterations
        #    lam - lambda parameter
        # Output:
        #    P - Final P (3x4) obtained after convergence
        
        # data normalization
        x, T = self.Normalize(x)
        X, U = self.Normalize(X)
        
        """your code here"""
        numPoint = len(x[0])
        s = T[0,0]
        P = T@P@np.linalg.inv(U)
        p = self.Parameterize(P)
        x_hat = np.dot(P, X)
    
        error_prev = self.Dehomogenize(x) - self.Dehomogenize(x_hat)
        error_prev = np.reshape(error_prev, (2*numPoint, 1), 'F')
        
        cost_prev = self.ComputeCostNormalized(P, self.Dehomogenize(x), self.Dehomogenize(X), s) #initialize of the cost 
        # you may modify this so long as the cost is computed
        # at each iteration
        i = 0
        while(i < max_iters): 
            J = self.Jacobian(self.Deparameterize(p),p,self.Dehomogenize(X))
            covariance = (s**2)* np.eye(2*numPoint)
            
            norm_Mat = np.dot(np.dot(np.transpose(J), np.linalg.inv(covariance)), J) + lam*np.eye(11)
            norm_vec = np.dot(np.dot(np.transpose(J), np.linalg.inv(covariance)), error_prev)
            delta = np.dot(np.linalg.inv(norm_Mat), norm_vec)
            p0 = p + delta
            P0 = self.Deparameterize(p0)
           
            cost0 = self.ComputeCostNormalized(P0, self.Dehomogenize(x), self.Dehomogenize(X), s)
            if cost0 > cost_prev: 
                lam = 10*lam
            else: 
                lam = 0.1*lam
                i += 1
                p = p0
                P = P0
                x_hat = np.dot(P, X)
                error_prev = self.Dehomogenize(x) - self.Dehomogenize(x_hat)
                error_prev = np.reshape(error_prev, (2*numPoint, 1), 'F')
    
                cost_prev = cost0
    
                print ('iter %03d Cost %.9f'%(i, cost_prev))
                    
        # data denormalization
        P = np.linalg.inv(T) @ P @ U
        return P


