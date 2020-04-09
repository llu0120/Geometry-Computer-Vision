from utility import *
from FeatureDetection import FeatureDetection
from FeatureMatching import FeatureMatching
from MSAC import MSAC
from LinearEstimate import LinearEstimate
from NonlinearLM import NonlinearLM

if __name__ == '__main__':
    """
    Feature Detection
    """
    FeatureDetection = FeatureDetection()
    # input images
    I1 = np.array(Image.open('img1.jpeg'), dtype='float')/255.
    I2 = np.array(Image.open('img2.jpeg'), dtype='float')/255.
    
    # parameters to tune
    w = 7
    t = 0.054
    w_nms = 7
    
    tic = time.time()
    
    # run feature detection algorithm on input images
    C1, pts1, J1_0, J1_1, J1_2 = FeatureDetection.RunFeatureDetection(I1, w, t, w_nms)
    C2, pts2, J2_0, J2_1, J2_2 = FeatureDetection.RunFeatureDetection(I2, w, t, w_nms)
    toc = time.time() - tic
    
    print('took %f secs'%toc)
    
    # display results
    plt.figure(figsize=(14,24))
    
    # show corners on original images
    ax = plt.subplot(1,2,1)
    plt.imshow(I1)
    for i in range(C1): # draw rectangles of size w around corners
        x,y = pts1[:,i]
        ax.add_patch(patches.Rectangle((x-w/2,y-w/2),w,w, fill=False))
    # plt.plot(pts1[0,:], pts1[1,:], '.b') # display subpixel corners
    plt.title('Found %d Corners'%C1)
    
    ax = plt.subplot(1,2,2)
    plt.imshow(I2)
    for i in range(C2):
        x,y = pts2[:,i]
        ax.add_patch(patches.Rectangle((x-w/2,y-w/2),w,w, fill=False))
    # plt.plot(pts2[0,:], pts2[1,:], '.b')
    plt.title('Found %d Corners'%C2)
    
    plt.show()
    
    """
    Feature Matching
    """
    FeatureMatching = FeatureMatching()
    # parameters to tune
    w = 11
    t = 0.78
    d = 0.8
    p = 170
    
    tic = time.time()
    # run the feature matching algorithm on the input images and detected features
    inds = FeatureMatching.RunFeatureMatching(I1, I2, pts1, pts2, w, t, d, p)
    toc = time.time() - tic
    
    print('took %f secs'%toc)
    
    # create new matrices of points which contain only the matched features 
    match1 = pts1[:,inds[0,:]]
    match2 = pts2[:,inds[1,:]]
    
    # # display the results
    plt.figure(figsize=(14,24))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    ax1.imshow(I1)
    ax2.imshow(I2)
    plt.title('Found %d Putative Matches'%match1.shape[1])
    for i in range(match1.shape[1]):
        x1,y1 = match1[:,i]
        x2,y2 = match2[:,i]
        ax1.plot([x1, x2],[y1, y2],'-r')
        ax1.add_patch(patches.Rectangle((x1-w/2,y1-w/2),w,w, fill=False))
        ax2.plot([x2, x1],[y2, y1],'-r')
        ax2.add_patch(patches.Rectangle((x2-w/2,y2-w/2),w,w, fill=False))
    
    plt.show()
    
    print('unique points in image 1: %d'%np.unique(inds[0,:]).shape[0])
    print('unique points in image 2: %d'%np.unique(inds[1,:]).shape[0])
    
    
    """
    Outlier Rejection by MSAC for Fundamental Matrix 
    """
    MSAC = MSAC()
    # MSAC parameters 
    thresh = 50
    tol = 3
    p = 0.99
    alpha = 0.95
    
    tic=time.time()
    cost_MSAC, F_MSAC, inliers, trials = MSAC.MSAC(match1, match2, thresh, tol, p, alpha)
    
    # choose just the inliers
    xin1 = match1[:,inliers]
    xin2 = match2[:,inliers]
    outliers = np.setdiff1d(np.arange(pts1.shape[1]),inliers)
    
    toc=time.time()
    time_total=toc-tic
    
    # display the results
    print('took %f secs'%time_total)
    print('%d iterations'%trials)
    print('inlier count: ',len(inliers))
    print('inliers: ',inliers)
    print('MSAC Cost = %.9f'%cost_MSAC)
    DisplayResults(F_MSAC, 'F_MSAC')
    
    # display the figures
    plt.figure(figsize=(14,8))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    ax1.imshow(I1)
    ax2.imshow(I2)
    
    for i in range(xin1.shape[1]):
        x1,y1 = xin1[:,i]
        x2,y2 = xin2[:,i]
        ax1.plot([x1, x2],[y1, y2],'-r')
        ax1.add_patch(patches.Rectangle((x1-w/2,y1-w/2),w,w, fill=False))
        ax2.plot([x2, x1],[y2, y1],'-r')
        ax2.add_patch(patches.Rectangle((x2-w/2,y2-w/2),w,w, fill=False))
    
    plt.show()
    
    """
    Linear Estimation of Fundamental Matrix by DLT 
    """
    LinearEstimate = LinearEstimate()
    # compute the linear estimate with data normalization
    print ('DLT with Data Normalization')
    time_start=time.time()
    F_DLT = LinearEstimate.DLT(xin1, xin2, normalize=True)
    time_total=time.time()-time_start
    
    # display the resulting F_DLT, scaled with its frobenius norm
    DisplayResults(F_DLT, 'F_DLT')
    
    """
    Nonlinear Optimization of Fundamental Matrix by LM 
    """
    NonlinearLM = NonlinearLM()
    # LM hyperparameters
    lam = .001
    max_iters = 100
    
    # Run LM initialized by DLT estimate
    print ('Sparse LM')
    time_start=time.time()
    
    # Uncomment the following lines to use sample inliers.
    #xin1 = np.loadtxt("hw5_pts1.txt").T
    #xin2 = np.loadtxt("hw5_pts2.txt").T
    
    x_scene, P_prime = NonlinearLM.scenePtInitialization(F_DLT, xin1, xin2)
    time_total1=time.time()-time_start
    time_start=time.time()
    F_LM = NonlinearLM.LM(F_DLT, xin1, xin2, x_scene, P_prime, max_iters, lam)
    time_total2=time.time()-time_start
    time_total = float(time_total1) + float(time_total2)
    print('took %f secs'%time_total)
    
    # display the resulting F_LM, scaled with its frobenius norm
    DisplayResults(F_LM, 'F_LM')
    
        