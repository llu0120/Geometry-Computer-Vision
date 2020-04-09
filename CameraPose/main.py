from MSAC import MSAC
from LinearEstimate import LinearEstimate
from NonlinearLM import NonlinearLM

if __name__ == '__main__':
    """
    Outlier Rejection from MSAC for Camera Pose Matrix
    """
    MSAC = MSAC()
    # load data
    x0=np.loadtxt('points2D.txt').T
    X0=np.loadtxt('points3D.txt').T
    print('x is', x0.shape)
    print('X is', X0.shape)
    
    K = np.array([[1545.0966799187809, 0, 639.5], 
          [0, 1545.0966799187809, 359.5], 
          [0, 0, 1]])
    
    print('K =')
    print(K)
    
    # MSAC parameters 
    thresh = 100
    tol = 5
    p = 0.99
    alpha = 0.95
    
    tic=time.time()
    
    cost_MSAC, P_MSAC, inliers, trials = MSAC.MSAC(x0, X0, K, thresh, tol, p, alpha)
    
    # choose just the inliers
    x = x0[:,inliers]
    X = X0[:,inliers]
    
    toc=time.time()
    time_total=toc-tic
    
    # display the results
    print('took %f secs'%time_total)
    print('%d iterations'%trials)
    print('inlier count: ',len(inliers))
    print('MSAC Cost=%.9f'%cost_MSAC)
    print('P = ')
    print(P_MSAC)
    print('inliers: ',inliers)

    """
    linear Estimation of Camera Pose Matrix by EPnP
    """
    LinearEstimate = LinearEstimate()
    tic=time.time()
    P_linear = LinearEstimate.EPnP(x, X, K)
    toc=time.time()
    time_total=toc-tic
    
    cost = LinearEstimate.ComputeCost(P_linear, x, X, K)
    print("cost", cost)
    
    # display the results
    print('took %f secs'%time_total)
    print('R_linear = ')
    print(P_linear[:,0:3])
    print('t_linear = ')
    print(P_linear[:,-1])
    
    '''
    Nonlinear Optimization of camera pose by Lavenberg-Marquardt (LM)
    '''
    NonlinearLM = NonlinearLM()
    # LM hyperparameters
    lam = .001
    max_iters = 100
    
    tic = time.time()
    P_LM = NonlinearLM.LM(P_linear, x, X, K, max_iters, lam)
    w_LM,_ = NonlinearLM.Parameterize(P_LM[:,0:3])
    toc = time.time()
    time_total = toc-tic
    
    # display the results
    print('took %f secs'%time_total)
    print('w_LM = ')
    print(w_LM)
    print('R_LM = ')
    print(P_LM[:,0:3])
    print('t_LM = ')
    print(P_LM[:,-1])