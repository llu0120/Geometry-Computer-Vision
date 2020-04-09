from utility import *

class FeatureDetection():
    
    def rgb2gray(self, rgb):
        """ Convert rgb image to grayscale.
        """
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def ImageGradient(self, I):
        # inputs: 
        # I is the input image (may be mxn for Grayscale or mxnx3 for RGB)
        #
        # outputs:
        # Ix is the derivative of the magnitude of the image w.r.t. x
        # Iy is the derivative of the magnitude of the image w.r.t. y
        
        """your code here"""
        I = self.rgb2gray(I)
        kernelX = np.array([[-1, 8, 0, -8, 1]])*0.5
        kernelY = np.transpose(kernelX)
        Ix = signal.convolve2d(I, kernelX, boundary='symm', mode='same')
        Iy = signal.convolve2d(I, kernelY, boundary='symm', mode='same')
        
        return Ix, Iy
      
    
    def MinorEigenvalueImage(self, Ix, Iy, w):
        # Calculate the minor eigenvalue image J
        #
        # inputs:
        # Ix is the derivative of the magnitude of the image w.r.t. x
        # Iy is the derivative of the magnitude of the image w.r.t. y
        # w is the size of the window used to compute the gradient matrix N
        #
        # outputs:
        # J0 is the mxn minor eigenvalue image of N before thresholding
    
        m, n = Ix.shape[:2]
        J0 = np.zeros((m,n))
    
        #Calculate your minor eigenvalue image J0.
        """your code here"""
        IxIx = Ix**2
        IxIy = Ix*Iy
        IyIy = Iy**2
        
        N = np.zeros((2,2,m,n)) #Gradient Matrix
        half_w = int(w/2)
        #Compute Minor Eigenvalue
        for i in range(w, m-w):
            for j in range(w, n-w): 
                #summation(average instead of summing) 
                N[0,0,i,j] = (1/(w**2))* np.sum(IxIx[i-half_w:i+half_w, j-half_w:j+half_w])
                N[0,1,i,j] = (1/(w**2))* np.sum(IxIy[i-half_w:i+half_w, j-half_w:j+half_w])
                N[1,0,i,j] = N[0,1,i,j]
                N[1,1,i,j] = (1/(w**2))* np.sum(IyIy[i-half_w:i+half_w, j-half_w:j+half_w])
                J0[i][j] = (np.trace(N[:,:,i,j])
                           - np.sqrt(np.trace(N[:,:,i,j])**2 - 4*np.linalg.det(N[:,:,i,j])))/2
        return J0, N
      
    def NMS(self, J, w_nms):
        # Apply nonmaximum supression to J using window w_nms
        #
        # inputs: 
        # J is the minor eigenvalue image input image after thresholding
        # w_nms is the size of the local nonmaximum suppression window
        # 
        # outputs:
        # J2 is the mxn resulting image after applying nonmaximum suppression
        # 
        
        J1 = J.copy()
        J2 = J.copy()
        """your code here"""
        m, n = J.shape
        half_w = int(w_nms/2)
        for i in range(m):
            for j in range(n):
                #up
                if i - half_w < 0: 
                    y_min = 0
                else: 
                    y_min = i - half_w
                
                #down 
                if i + half_w > m: 
                    y_max = m 
                else: 
                    y_max = i + half_w
                    
                #left 
                if j - half_w < 0: 
                    x_min = 0
                else: 
                    x_min = j - half_w 
                
                #right 
                if j + half_w > n: 
                    x_max = n
                else: 
                    x_max = j + half_w 
                    
                J1[i][j] = np.amax(J2[y_min:y_max, x_min:x_max])
                
        for i in range(m):
            for j in range(n):
                if J2[i][j] < J1[i][j]:
                    J2[i][j] = 0 
        
        return J2
    
    
    def ForstnerCornerDetector(self, Ix, Iy, w, t, w_nms):
        # Calculate the minor eigenvalue image J
        # Threshold J
        # Run non-maxima suppression on the thresholded J
        # Gather the coordinates of the nonzero pixels in J 
        # Then compute the sub pixel location of each point using the Forstner operator
        #
        # inputs:
        # Ix is the derivative of the magnitude of the image w.r.t. x
        # Iy is the derivative of the magnitude of the image w.r.t. y
        # w is the size of the window used to compute the gradient matrix N
        # t is the minor eigenvalue threshold
        # w_nms is the size of the local nonmaximum suppression window
        #
        # outputs:
        # C is the number of corners detected in each image
        # pts is the 2xC array of coordinates of subpixel accurate corners
        #     found using the Forstner corner detector
        # J0 is the mxn minor eigenvalue image of N before thresholding
        # J1 is the mxn minor eigenvalue image of N after thresholding
        # J2 is the mxn minor eigenvalue image of N after thresholding and NMS
    
        m, n = Ix.shape
        #Calculate your minor eigenvalue image J0 and its thresholded version J1.
        """your code here"""
        half_w = int(w/2)
        J0, N = self.MinorEigenvalueImage(Ix, Iy, w)
        
        J1 = J0.copy()
        J1[J1 < t] = 0      
        
        #Run non-maxima suppression on your thresholded minor eigenvalue image.
        J2 = self.NMS(J1, w_nms)
        
        #Detect corners.
        """your code here"""
        
        b1 = np.zeros((m,n))
        b2 = np.zeros((m,n))
        for y in range(m):
            for x in range(n):
                b1[y][x] = x*Ix[y][x]**2 + y*Ix[y][x]*Iy[y][x]
                b2[y][x] = x*Ix[y][x]*Iy[y][x] + y*Iy[y][x]**2
        
        C = 0
        pts = []
        b = np.zeros((2,1,m,n))
        for i in range(half_w, m-half_w):
            for j in range(half_w, n-half_w):
                #Summation(average instead of summing)
                b[0,0,i,j] = (1/(w**2))* np.sum(b1[i-half_w:i+half_w, j-half_w:j+half_w])
                b[1,0,i,j] = (1/(w**2))* np.sum(b2[i-half_w:i+half_w, j-half_w:j+half_w])
                
                if J2[i][j] > 0:
                    C += 1
                    cor = np.linalg.inv(N[:,:,i,j]).dot(b[:,:,i,j])
                    xcor, ycor = cor[0][0], cor[1][0]
                    pts.append([xcor, ycor])
        pts = np.transpose(pts)
        print(pts.shape)
        print(C)
        return C, pts, J0, J1, J2
    
    
    # feature detection
    def RunFeatureDetection(self, I, w, t, w_nms):
        Ix, Iy = self.ImageGradient(I)
        C, pts, J0, J1, J2 = self.ForstnerCornerDetector(Ix, Iy, w, t, w_nms)
        return C, pts, J0, J1, J2
