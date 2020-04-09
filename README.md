# Geometry-Computer-Vision
## One View Geometry 
1. Camera Projection Matrix (Uncalibrated Case) 

	a. Linear Estimation by Direct Linear Transform (DLT)
	
	b. Nonlinear Optimization by Dense Lavenberg-Marquardt (LM) 
2. Camera Pose Matrix (Calibrated Case) 

	a. Outlier Rejection by MSAC by Finswelder Solution to calculate Camera Pose Matrix 
	
	b. Linear Estimation by EPnP
	
	c. Nonlinear Estimation by Dense Lavenberg-Marquardt (LM)
 
## Two View Geometry
1. Planar Projective Transformation 

	a. Feature Detection by Shi-Tomashi Corner Detection Algorithm 
	
	b. Feature Detection by One-to-One Naive Matching Algorithm 
	
	c. Outlier Rejection by 4-Point algorithm to calculate Homography matrix
	
	d. Linear Estimation by Direct Linear Transofrom (DLT) 
	
	e. Nonlinear Estimation by Sparse Lavenberg-Marquardt (LM) jointly optimized scence points 
2. Fundamental Matrix (Uncalibrated Case) 

	a. Feature Detection by Shi-Tomashi Corner Detection Algorithm
	
	b. Feature Detection by One-to-One Naive Matching Algorithm 

	c. Outlier Rejection by 7-Point algorithm to calculate Fundamental matrix

	d. Linear Estimation by Direct Linear Transofrom (DLT)

	e. Nonlinear Estimation by Sparse Lavenberg-Marquardt (LM) jointly optimized 3D scence points(Derived by Triangulation)  
