'''
Outlier Rejection for camera pose estimation 
'''
from utility import *

class MSAC(): 
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
    
    '''
    3-point algorithm of Finsterwalder to estimate camera pose 
    '''
    def computeModel(self, X1, X2, X3, x1, x2, x3, K, X, x): 
        # Inputs:
        #    X1, X2, X3 - 3D inhomogeneous scene points
        #    x1, x2, x3 - 2D inhomogeneous image points
        #    K - camera calibration matrix
        #    X - 3D inhomogeneous scene points
        #    x - 2D inhomogeneous image points
        # Output:
        #    P - camera projection matrix P estimated by Finsterwalder's Solution 
        P = np.zeros((3,4))
        f = K[1,1]
        
        #Compute Distance between 3D points
        a = self.computeDistance(X2, X3)
        b = self.computeDistance(X1, X3)
        c = self.computeDistance(X1, X2)
        
        #make the image points into normalized coordinate
        x1_head = np.linalg.inv(K)@self.Homogenize(x1)
        x2_head = np.linalg.inv(K)@self.Homogenize(x2)
        x3_head = np.linalg.inv(K)@self.Homogenize(x3)
        
        #Compute ui, vi 
        u1 = f * x1_head[0,0] / x1_head[2,0]
        v1 = f * x1_head[1,0] / x1_head[2,0] 
        
        u2 = f * x2_head[0,0] / x2_head[2,0]
        v2 = f * x2_head[1,0] / x2_head[2,0] 
        
        u3 = f * x3_head[0,0] / x3_head[2,0]
        v3 = f * x3_head[1,0] / x3_head[2,0]
        
        #Compute j1, j2, j3 from ui, vi 
        j1 = (1 / np.sqrt(u1**2+v1**2+f**2)) * np.array([[u1],[v1],[f]]) 
        j2 = (1 / np.sqrt(u2**2+v2**2+f**2)) * np.array([[u2],[v2],[f]])
        j3 = (1 / np.sqrt(u3**2+v3**2+f**2)) * np.array([[u3],[v3],[f]])
        
        #Compute alpha, beta, gamma angle from j1, j2, j3 
        cosAlpha = float(np.dot(np.transpose(j2), j3))
        cosBeta = float(np.dot(np.transpose(j1), j3))
        cosGamma = float(np.dot(np.transpose(j1), j2))
        
        sinAlpha_sqaure = 1 - cosAlpha**2 
        sinBeta_sqaure = 1 - cosBeta**2 
        sinGamma_sqaure = 1 - cosGamma**2 
        
        #Solve for lamda 
        G = c**2 * (c**2 * sinBeta_sqaure - b**2 * sinGamma_sqaure)
        H = b**2 * (b**2 - a**2) * sinGamma_sqaure + c**2 * (c**2 + 2 * a**2) * sinBeta_sqaure + \
            2 * b**2 * c**2 * (-1 + cosAlpha * cosBeta * cosGamma)
        I = b**2 * (b**2 - c**2) * sinAlpha_sqaure + a**2 * (a**2 + 2 * c**2) * sinBeta_sqaure + \
            2 * a**2 * b**2 * (-1 + cosAlpha * cosBeta * cosGamma)
        J = a**2 * (a**2 * sinBeta_sqaure - b**2 * sinAlpha_sqaure)
    
        lamda = np.roots([G,H,I,J])
        for i in range(len(lamda)):
            if np.iscomplex(lamda[i]) == False: 
                lamda0 = np.real(lamda[i])
    
        #Compute A, B, C, D, E, F
        A = 1 + lamda0
        B = -cosAlpha
        C = ((b**2 - a**2) / b**2) - lamda0 * c**2 / b**2
        D = -lamda0 * cosGamma
        E = ((a**2 / b**2) + lamda0 * c**2 / b**2) * cosBeta
        F = (-a**2 / b**2) + lamda0 * ((b**2 - c**2) / b**2)
        
        if (B**2 - A * C) >= 0 and E**2 - C * F >= 0:
            #Compute p, q
            p = np.sqrt(B**2 - A * C)
            q = np.sign(B * E - C * D) * np.sqrt(E**2 - C * F)
    
            #Compute two sets of m, n 
            m1 = (-B + p) / C
            n1 = (-(E - q)) / C
    
            m2 = (-B - p) / C
            n2 = (-(E + q)) / C
    
            #Use two sets of m, n to compute two sets of A, B, C 
            A1 = b**2 - m1**2 * c**2 
            B1 = c**2 * (cosBeta - n1) * m1 - b**2 * cosGamma
            C1 = -c**2 * n1**2 + 2 * c**2 * n1 * cosBeta + b**2 - c** 2
    
            A2 = b**2 - m2**2 * c**2 
            B2 = c**2 * (cosBeta - n2) * m2 - b**2 * cosGamma
            C2 = -c**2 * n2**2 + 2 * c**2 * n2 * cosBeta + b**2 - c** 2
    
            #Derive 4 pairs of u and v 
            u = []
            v = []
            u_large1 = (-np.sign(B1) / A1) * (abs(B1) + np.sqrt(B1**2 - A1 * C1))    
            u_small1 = C1 / (A1 * u_large1)
            v_large1 = u_large1 * m1 + n1
            v_small1 = u_small1 * m1 + n1
    
            u.append(u_large1)
            u.append(u_small1)
            v.append(v_large1)
            v.append(v_small1)
    
            u_large2 = (-np.sign(B2) / A2) * (abs(B2) + np.sqrt(B2**2 - A2 * C2))    
            u_small2 = C2 / (A2 * u_large2)
            v_large2 = u_large2 * m2 + n2
            v_small2 = u_small2 * m2 + n2
    
            u.append(u_large2)
            u.append(u_small2)
            v.append(v_large2)
            v.append(v_small2)
        else: 
            return P 
    
        #Pick one pair of u and to compute P
        cost_optimal = np.inf
        for i in range(len(u)):
            u_tmp = u[i]
            v_tmp = v[i]
            
            u_check1 = np.iscomplex(u_tmp)
            v_check1 = np.iscomplex(v_tmp)
            if u_check1 == True or v_check1 == True: 
                continue
                
            u_check2 = np.isnan(u_tmp)
            v_check2 = np.isnan(v_tmp)
            if u_check2 == True or v_check2 == True: 
                continue
                
            if u_tmp < 0 or v_tmp < 0: 
                continue
            
            #Compute s1, s2, s3
            s1 = np.sqrt(c**2 / (1 + u_tmp**2 - 2 * u_tmp * cosGamma))
            s2 = u_tmp * s1 
            s3 = v_tmp * s1 
            
            ######### Iterative Closest Point ###########
            #3D points in camera coordinate
            X1_cam = s1 * j1
            X2_cam = s2 * j2
            X3_cam = s3 * j3
        
            mean_X = (X1 + X2 + X3) / 3 
            mean_X_cam = (X1_cam + X2_cam + X3_cam) / 3
            
            q1 = X1 - mean_X
            q2 = X2 - mean_X
            q3 = X3 - mean_X
            
            q1_cam = X1_cam - mean_X_cam
            q2_cam = X2_cam - mean_X_cam
            q3_cam = X3_cam - mean_X_cam
            
            w = np.dot(q1_cam, np.transpose(q1)) + np.dot(q2_cam, np.transpose(q2)) + np.dot(q3_cam, np.transpose(q3))
            U, _, V = np.linalg.svd(w)
            if (np.linalg.det(U)*np.linalg.det(np.transpose(V)) < 0):
                S = np.eye(3)
                S[2,2] = -1
                R = np.dot(np.dot(U, S), V)
            else:
                R = np.dot(U, V)
            t = mean_X_cam - np.dot(R, mean_X)        
            P_tmp = np.concatenate((R, t), axis=1)
            
            #Compute cost
            cost, _ = self.computeCostMSAC(P_tmp, x, X, K, tol=None)
            if cost < cost_optimal:
                P = P_tmp
                cost_optimal = cost
        
        return P
    
    def computeDistance(self, X1, X2):
        # Inputs:
        #    X1, X2 - Two 3D inhomogeneous scene point2
        # Output:
        #    distance 
        return float(abs(np.sqrt((X1[0] - X2[0])**2 + (X1[1] - X2[1])**2 + (X1[2] - X2[2])**2)))
    
    def computeError(self, X, x, K, P):
        # Inputs:
        #    X - a 3D inhomogeneous scene point
        #    x - a 2D inhomogeneous image point
        #    K - camera calibration matrix
        #    P - camera projection matrix
        # Output:
        #    error - total projection error
        x_head = K@P@self.Homogenize(X)
        x_head = self.Dehomogenize(x_head)
        error = np.linalg.norm(x_head - x)
        
        return error
            
    def computeCostMSAC(self, P, x, X, K, tol=None):
        # Inputs:
        #    P - camera projection matrix
        #    x - 2D inhomogeneous image points
        #    X - 3D inhomogeneous scene points
        #    K - camera calibration matrix
        #    tol - reprojection error tolerance 
        # Output:
        #    cost - total projection error
        #    inliers_tmp - number of inliers to update maxtrials 
        inliers_tmp = 0 
        cost = 0
        numPoint = X.shape[1] 
        for i in range(numPoint):
            #compute error
            error = self.computeError(X[:,i:i+1], x[:,i:i+1], K, P)
            if tol != None:
                if error < tol:
                    cost += error
                    inliers_tmp += 1
                else:
                    cost += tol
            else: 
                cost += error
        
        return cost, inliers_tmp 
                
    def MSAC(self, x, X, K, thresh, tol, p, alpha):
        # Inputs:
        #    x - 2D inhomogeneous image points
        #    X - 3D inhomogeneous scene points
        #    K - camera calibration matrix
        #    thresh - cost threshold
        #    tol - reprojection error tolerance 
        #    p - probability that as least one of the random samples does not contain any outliers   
        #    alpha - probability of a data point is an inlier
        #
        # Output:
        #    consensus_min_cost - final cost from MSAC
        #    consensus_min_cost_model - camera projection matrix P
        #    inliers - list of indices of the inliers corresponding to input data
        #    trials - number of attempts taken to find consensus set
        
        """your code here"""
        numPoint = np.shape(X)[1]
        trials = 0
        max_trials = np.inf
        consensus_min_cost = np.inf
        consensus_min_cost_model = np.zeros((3,4))
        while max_trials > trials and consensus_min_cost > thresh: 
            #Random pick three index for Finsterwalder's Solution 
            randomInd = random.sample(range(0, numPoint), 3)
            X1 = X[:,randomInd[0]:randomInd[0]+1]
            x1 = x[:,randomInd[0]:randomInd[0]+1]
            
            X2 = X[:,randomInd[1]:randomInd[1]+1]
            x2 = x[:,randomInd[1]:randomInd[1]+1]
            
            X3 = X[:,randomInd[2]:randomInd[2]+1]
            x3 = x[:,randomInd[2]:randomInd[2]+1]
            
            #Compute camera projection matrix 
            P_model = self.computeModel(X1, X2, X3, x1, x2, x3, K, X, x)
            
            #Compute error --> cost 
            cost, inliers_tmp = self.computeCostMSAC(P_model, x, X, K, tol)
            
            #Update model and cost value 
            if (cost < consensus_min_cost):
                consensus_min_cost = cost 
                consensus_min_cost_model = P_model
                
            #Update max_trials 
            if (inliers_tmp != 0):
                w = inliers_tmp / numPoint
                max_trials = np.log(1 - p) / np.log(1 - w**3)
            
            variance = 1
            thresh = chi2.ppf(alpha, df=2) * variance**2
            trials += 1
    
        #Compute number of inliers
        inliers = []
        for i in range(numPoint): 
            error = self.computeError(X[:,i:i+1], x[:,i:i+1], K, consensus_min_cost_model)
            if (error <= tol):
                inliers.append(i)
                
        return consensus_min_cost, consensus_min_cost_model, inliers, trials
    
    