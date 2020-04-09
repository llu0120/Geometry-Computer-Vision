from utility import *
'''
Feature Matching by one by one naive matching
'''

class FeatureMatching():
    def rgb2gray(self, rgb):
        """ Convert rgb image to grayscale.
        """
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def NCC_score(self, I1, I2, c1, c2, w, p):
        #inputs: 
        #I1, I2 are the input images 
        #c1, c2 are the center of in the windows 
        #w is the size of the matching window to compute correlation coefficients
        
        m, n = I1.shape
        y1_min, y1_max = int(c1[1])-w//2, int(c1[1])+w//2+1
        x1_min, x1_max = int(c1[0])-w//2, int(c1[0])+w//2+1
        y2_min, y2_max = int(c2[1])-w//2, int(c2[1])+w//2+1
        x2_min, x2_max = int(c2[0])-w//2, int(c2[0])+w//2+1
        if y1_min < 0 or y1_max > m or x1_min < 0 or x1_max > n or y2_min < 0 or y2_max > m or x2_min < 0 or x2_max > n:
            return 0 
        if np.sqrt(np.sum((c1 - c2)**2)) > p: 
            return 0
        window_area_img1 = I1[y1_min:y1_max, x1_min:x1_max]
        w_mean_img1 = np.mean(window_area_img1)
        ssd_img1 = np.sum((window_area_img1 - w_mean_img1)**2) 
        ncc_img1 = (window_area_img1 - w_mean_img1) / np.sqrt(ssd_img1)
        window_area_img2 = I2[y2_min:y2_max, x2_min:x2_max] 
        w_mean_img2 = np.mean(window_area_img2)
        ssd_img2 = np.sum((window_area_img2 - w_mean_img2)**2) 
        ncc_img2 = (window_area_img2 - w_mean_img2) / np.sqrt(ssd_img2)
    
        score = np.sum(np.multiply(ncc_img1, ncc_img2))
        return score
    
    def NCC(self, I1, I2, pts1, pts2, w, p):
        # compute the normalized cross correlation between image patches I1, I2
        # result should be in the range [-1,1]
        #
        # inputs:
        # I1, I2 are the input images
        # pts1, pts2 are the point to be matched
        # w is the size of the matching window to compute correlation coefficients
        #
        # output:
        # normalized cross correlation matrix of scores between all windows in 
        #    image 1 and all windows in image 2
        
        """your code here"""
        I1 = self.rgb2gray(I1)
        I2 = self.rgb2gray(I2)
        cor1_num = pts1.shape[1]
        cor2_num = pts2.shape[1]
        
        scores = np.zeros((cor1_num, cor2_num))
        pts1 = np.transpose(pts1) #cor1_num x 2
        pts2 = np.transpose(pts2) #cor2_num x 2
        #Construct correlation array 
        for i in range(cor1_num): 
            cor1 = pts1[i]
            for j in range(cor2_num):
                cor2 = pts2[j]
                scores[i][j] = self.NCC_score(I1, I2, cor1, cor2, w, p)
        return scores
    
    
    def Match(self, scores, t, d, p):
        # perform the one-to-one correspondence matching on the correlation coefficient matrix
        # 
        # inputs:
        # scores is the NCC matrix
        # t is the correlation coefficient threshold
        # d distance ration threshold
        # p is the size of the proximity window
        #
        # output:
        # list of the feature coordinates in image 1 and image 2 
        
        """your code here"""
        inds = []
        m, n = scores.shape
        mask = np.ones((m, n))
        
        #One to One matching 
        max_val = np.amax(scores)
        while (max_val > t):
            correlation_array_masked = np.multiply(scores, mask)  
            index = unravel_index(correlation_array_masked.argmax(), correlation_array_masked.shape)
            max_val = np.max(correlation_array_masked)
            correlation_array_masked[index] = -1
            scores[index] = -1
            
            row = index[0]
            col = index[1]
            same_row = scores[row]
            same_col = [x[col] for x in correlation_array_masked]
            
            next_max_val = max(np.max(same_row), np.max(same_col))
    
            correlation_array_masked[index] = max_val 
            scores[index] = max_val 
            if (1 - max_val) < (1 - next_max_val)*d:
                inds.append([int(index[0]), int(index[1])])
                mask[row,:] = 0 
                mask[:,col] = 0 
            else:
                mask[index] = 0 
        inds = np.transpose(inds)
        return inds

    
    def RunFeatureMatching(self, I1, I2, pts1, pts2, w, t, d, p):
        # inputs:
        # I1, I2 are the input images
        # pts1, pts2 are the point to be matched
        # w is the size of the matching window to compute correlation coefficients
        # t is the correlation coefficient threshold
        # d distance ration threshold
        # p is the size of the proximity window
        #
        # outputs:
        # inds is a 2xk matrix of matches where inds[0,i] indexs a point pts1 
        #     and inds[1,i] indexs a point in pts2, where k is the number of matches
        
        scores = self.NCC(I1, I2, pts1, pts2, w, p)
        inds = self.Match(scores, t, d, p)
        return inds
    
    