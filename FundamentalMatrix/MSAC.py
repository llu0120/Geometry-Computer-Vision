from utility import *

class MSAC():
    
    def Homogenize(self, x):
        # converts points from inhomogeneous to homogeneous coordinates
        return np.vstack((x,np.ones((1,x.shape[1]))))
    
    def Dehomogenize(self, x):
        # converts points from homogeneous to inhomogeneous coordinates
        return x[:-1]/x[-1]
    
    def computeModel(self, pts1, pts2): #7-point algorithm
        # Inputs:
        #    pts1 - matched feature correspondences in image 1 2x7
        #    pts2 - matched feature correspondences in image 2 2x7
        #
        # Output:
        #    F - Fundamental Matrix 3 x 3
        pts1 = self.Homogenize(pts1)
        pts2 = self.Homogenize(pts2)
        A = np.zeros((7, 9))
        for i in range(7):
            ai = np.kron(np.transpose(pts2[:, i:i+1]), np.transpose(pts1[:, i:i+1])) 
            A[i] = ai
    
        U, S, V = np.linalg.svd(A)
        a = V[8:9, :]
        b = V[7:8, :]
        
        F1 = np.reshape(a, (3,3))
        F2 = np.reshape(b, (3,3))
        alpha = Symbol('alpha')
        F = Matrix(alpha*F1 + F2)
        answer = solve(Matrix.det(F), alpha)
        alpha_sol = np.array(answer).astype(np.complex64)   
        alpha_sol = np.real(alpha_sol)
    
        return F1, F2, alpha_sol 
    
    def computeError(self, pt1, pt2, F): #(squared) Sampson error
        # Inputs:
        #    pt1 - a 2D inhomogeneous image1 point
        #    pt2 - a 2D inhomogeneous image2 point
        #
        # Output:
        #    error - Sampson error
        pt1 = self.Homogenize(pt1)
        pt2 = self.Homogenize(pt2)
        upper = (np.transpose(pt2)@F@pt1)**2
        lower = ((np.transpose(pt2)@F[:, 0:1])**2 + (np.transpose(pt2)@F[:, 1:2])**2 
                + (F[0:1, :]@pt1)**2 + (F[1:2, :]@pt1)**2)
        
        error = upper/lower
        return error 
    
    def computeCostMSAC(self, F, pts1, pts2, tol=None):
        # Inputs:
        #    F - Fundamental Matrix
        #    pts1 - 2D inhomogeneous image1 points
        #    pts2 - 2D inhomogeneous image2 points
        #    tol - Sampson error tolerance 
        #
        # Output:
        #    cost - total Sampson error
        #    inliers_tmp - number of inliers to update maxtrials 
        inliers_tmp = 0 
        cost = 0
        numPoint = pts1.shape[1] 
        for i in range(numPoint):
            #compute error
            error = self.computeError(pts1[:,i:i+1], pts2[:,i:i+1], F)
            if tol != None:
                if error < tol:
                    cost += error
                    inliers_tmp += 1
                else:
                    cost += tol
            else: 
                cost += error
        
        return cost, inliers_tmp 
    
    def MSAC(self, pts1, pts2, thresh, tol, p, alpha):
        # Inputs:
        #    pts1 - matched feature correspondences in image 1
        #    pts2 - matched feature correspondences in image 2
        #    thresh - cost threshold
        #    tol - reprojection error tolerance 
        #    p - probability that as least one of the random samples does not contain any outliers   
        #    alpha - probability of a data point is an inlier
        #
        # Output:
        #    consensus_min_cost - final cost from MSAC
        #    consensus_min_cost_model - fundamental matrix F
        #    inliers - list of indices of the inliers corresponding to input data
        #    trials - number of attempts taken to find consensus set
        
        """your code here"""
        numPoint = np.shape(pts1)[1]
        trials = 0
        max_trials = 100
        consensus_min_cost = np.inf
        consensus_min_cost_model = np.zeros((3,3))
        random.seed(1)
        while max_trials > trials and consensus_min_cost > thresh: 
            #Random pick Seven index for 7-point algorithm
            point_ind = []
            pts1_tmp = np.zeros((2, 7))
            pts2_tmp = np.zeros((2, 7))
            
            random_ind = random.sample(range(0, numPoint), 7)    
            for i in range(7):
                pts1_tmp[:, i:i+1] = pts1[:, random_ind[i]:random_ind[i]+1]
                pts2_tmp[:, i:i+1] = pts2[:, random_ind[i]:random_ind[i]+1]
                
            #Compute Fundamental matrix 
            F1, F2, alpha_sol = self.computeModel(pts1_tmp, pts2_tmp)
            cost_prev_tmp = np.inf
    
            for i in range(3):
                F_tmp = alpha_sol[i]*F1 + F2
                cost_tmp, _ = self.computeCostMSAC(F_tmp, pts1, pts2)
                if cost_tmp < cost_prev_tmp: 
                    F_best_tmp = F_tmp 
                    cost_prev_tmp = cost_tmp
            
            F_model = F_best_tmp
            
            #Compute error --> cost 
            cost, inliers_tmp = self.computeCostMSAC(F_model, pts1, pts2, tol)
            
            #Update model and cost value 
            if (cost < consensus_min_cost):
                consensus_min_cost = cost 
                consensus_min_cost_model = F_model
                
            #Update max_trials 
            if (inliers_tmp != 0):
                w = inliers_tmp / numPoint
                max_trials = np.log(1 - p) / np.log(1 - w**7)
            
            variance = 1
            thresh = chi2.ppf(alpha, df=2) * variance**2
            trials += 1
    
        #Compute number of inliers
        inliers = []
        for i in range(numPoint): 
            error = self.computeError(pts1[:,i:i+1], pts2[:,i:i+1], consensus_min_cost_model)
            if (error <= tol):
                inliers.append(i)
                
        return consensus_min_cost, consensus_min_cost_model, inliers, trials


