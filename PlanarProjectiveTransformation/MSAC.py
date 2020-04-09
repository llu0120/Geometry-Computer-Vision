from utility import *

'''
MSAC Outlier Rejection 
'''

class MSAC(): 
    def Homogenize(self, x):
        # converts points from inhomogeneous to homogeneous coordinates
        return np.vstack((x,np.ones((1,x.shape[1]))))

    def Dehomogenize(self, x):
        # converts points from homogeneous to inhomogeneous coordinates
        return x[:-1]/x[-1]
    
    '''
    4-point algorithm
    '''
    def computeModel(self, pts1, pts2): #4-point algorithm
        # Inputs:
        #    pts1 - matched feature correspondences in image 1 2x4
        #    pts2 - matched feature correspondences in image 2 2x4
        #
        # Output:
        #    H - planar projective transformation from the 2D points in image 1 to the 2D points in image 2 3x3
        pts1 = self.Homogenize(pts1)
        pts2 = self.Homogenize(pts2)
        pts1_threeCol = pts1[:,0:3]
        pts1_fourthCol = pts1[:, 3:4]
    
        pts2_threeCol = pts2[:, 0:3]
        pts2_fourthCol = pts2[:, 3:4]
        
        lamda_pts1 = np.dot(np.linalg.inv(pts1_threeCol), pts1_fourthCol)
        lamda_pts2 = np.dot(np.linalg.inv(pts2_threeCol), pts2_fourthCol)
    
        H_pts1_inv = pts1_threeCol * np.transpose(lamda_pts1)
        H_pts2_inv = pts2_threeCol * np.transpose(lamda_pts2)
        
        H = np.dot(H_pts2_inv, np.linalg.inv(H_pts1_inv))
        return H
    
    def computeError(self, pt1, pt2, H): #(squared) Sampson error
        # Inputs:
        #    pt1 - a 2D inhomogeneous image1 point
        #    pt2 - a 2D inhomogeneous image2 point
        #
        # Output:
        #    error - Sampson error
        h11, h12, h13 = H[0][0], H[0][1], H[0][2]
        h21, h22, h23 = H[1][0], H[1][1], H[1][2]
        h31, h32, h33 = H[2][0], H[2][1], H[2][2]
        
        xi = float(pt1[0])
        yi = float(pt1[1])
        xi_prime = float(pt2[0])
        yi_prime = float(pt2[1])
        J = np.array([[-h21+yi_prime*h31, -h22+yi_prime*h32, 0, xi*h31+yi*h32+h33],
                      [h11-xi_prime*h31, h12-xi_prime*h32, -(xi*h31+yi*h32+h33), 0]])
        
        epslon = np.array([[-(xi*h21 + yi*h22 + h23) + yi_prime*(xi*h31 + yi*h32 + h33)],
                           [xi*h11 + yi*h12 + h13 - xi_prime*(xi*h31 + yi*h32 + h33)]])
        error = np.dot(np.transpose(epslon), np.dot(np.dot(J, np.transpose(J)), epslon))
        return error 
    
    def computeCostMSAC(self, H, pts1, pts2, tol=None):
        # Inputs:
        #    H - camera projection matrix
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
            error = self.computeError(pts1[:,i:i+1], pts2[:,i:i+1], H)
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
        #
        # Output:
        #    consensus_min_cost - final cost from MSAC
        #    consensus_min_cost_model - planar projective transformation matrix H
        #    inliers - list of indices of the inliers corresponding to input data
        #    trials - number of attempts taken to find consensus set
        
        """your code here"""
        numPoint = np.shape(pts1)[1]
        trials = 0
        max_trials = 100
        consensus_min_cost = np.inf
        consensus_min_cost_model = np.zeros((3,3))
        while max_trials > trials and consensus_min_cost > thresh: 
            #Random pick Four index for 4-point algorithm
            randomInd = random.sample(range(0, numPoint), 4)
            pt1_1 = pts1[:,randomInd[0]:randomInd[0]+1]
            pt2_1 = pts2[:,randomInd[0]:randomInd[0]+1]
            
            pt1_2 = pts1[:,randomInd[1]:randomInd[1]+1]
            pt2_2 = pts2[:,randomInd[1]:randomInd[1]+1]
            
            pt1_3 = pts1[:,randomInd[2]:randomInd[2]+1]
            pt2_3 = pts2[:,randomInd[2]:randomInd[2]+1]
            
            pt1_4 = pts1[:,randomInd[3]:randomInd[3]+1]
            pt2_4 = pts2[:,randomInd[3]:randomInd[3]+1]
            
            pts1_tmp1 = np.concatenate((pt1_1, pt1_2), axis=1)
            pts1_tmp2 = np.concatenate((pt1_3, pt1_4), axis=1)
            pts1_tmp = np.concatenate((pts1_tmp1, pts1_tmp2), axis=1)
            
            pts2_tmp1 = np.concatenate((pt2_1, pt2_2), axis=1)
            pts2_tmp2 = np.concatenate((pt2_3, pt2_4), axis=1)
            pts2_tmp = np.concatenate((pts2_tmp1, pts2_tmp2), axis=1)
            
            
            #Compute Homography matrix 
            H_model = self.computeModel(pts1_tmp, pts2_tmp)
            
            #Compute error --> cost 
            cost, inliers_tmp = self.computeCostMSAC(H_model, pts1, pts2, tol)
            
            #Update model and cost value 
            if (cost < consensus_min_cost):
                consensus_min_cost = cost 
                consensus_min_cost_model = H_model
                
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
            error = self.computeError(pts1[:,i:i+1], pts2[:,i:i+1], consensus_min_cost_model)
            if (error <= tol):
                inliers.append(i)
                
        return consensus_min_cost, consensus_min_cost_model, inliers, trials
    
    
    