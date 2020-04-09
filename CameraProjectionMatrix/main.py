from utility import *
from LinearEstimate import LinearEstimate
from NonlinearLM import NonlinearLM

if __name__ == '__main__':
    """
    Linear Estimation of Camera Projection Matrix by DLT
    """
    LinearEstimate = LinearEstimate()
    # load the data
    x=np.loadtxt('points2D.txt').T
    X=np.loadtxt('points3D.txt').T
    
    # compute the linear estimate without data normalization
    print ('Running DLT without data normalization')
    time_start=time.time()
    P_DLT = LinearEstimate.DLT(x, X, normalize=False)
    cost = LinearEstimate.ComputeCost(P_DLT, x, X)
    time_total=time.time()-time_start
    # display the results
    print('took %f secs'%time_total)
    print('Cost=%.9f'%cost)
    
    
    # compute the linear estimate with data normalization
    print ('Running DLT with data normalization')
    time_start=time.time()
    P_DLT = LinearEstimate.DLT(x, X, normalize=True)
    cost = LinearEstimate.ComputeCost(P_DLT, x, X)
    time_total=time.time()-time_start
    # display the results
    print('took %f secs'%time_total)
    print('Cost=%.9f'%cost)
    
    """
    Nonlinear Optimization of Camera Projection Matrix by LM
    """
    NonlinearLM = NonlinearLM()
    # LM hyperparameters
    lam = .001
    max_iters = 100
    
    # Run LM initialized by DLT estimate with data normalization
    print ('Running LM with data normalization')
    print ('iter %03d Cost %.9f'%(0, cost))
    time_start=time.time()
    P_LM = NonlinearLM.LM(P_DLT, x, X, max_iters, lam)
    time_total=time.time()-time_start
    print('took %f secs'%time_total)
    
    #Linear Estimate as a intial value 
    displayResults(P_DLT, x, X, 'P_DLT')
    
    #Nonlinear Optimization by LM 
    displayResults(P_LM, x, X, 'P_LM')
