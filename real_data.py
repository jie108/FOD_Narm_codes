import numpy as np

# Gauss-Newton algorithm (check wiki of Levenberg–Marquardt algorithm for more details)
# objective function S(beta)=||y-exp(-X*beta)||^2=||y-exp(-bvec^T*D*bvec)||^2, where f(beta)=exp(-X*beta)
# Jacobian matrix J=\partial{exp(-X*beta)}/\partial{beta}=-X.*exp(-X*beta), 
# where .* multiplies each row of X by corresponding element in exp(-X*beta)
# increment delta satisfy (J^T*J)*delta=J^T*(y-exp(-X*beta))
# reason to re-parameterize D&bvec by beta&X is to ensure D is symmetric, 
# such that there are 6 parameters to estimate instead of 9 in D
def gauss_newton_dwi(bvec, bval, y, thresh = 1e-10):

	n = bvec.shape[1]
	X = np.zeros((n, 6))
	X[:, :3] = (bvec**2).T
	X[:, 3], X[:, 4], X[:, 5] = 2*bvec[0]*bvec[1], 2*bvec[0]*bvec[2], 2*bvec[1]*bvec[2]
	X = (X.T*bval).T

	beta = -np.linalg.solve(X.T.dot(X), X.T.dot(np.log(y)))  # initialization of beta (log(y)\approx -X*beta)
	delta = beta.copy()

	while np.linalg.norm(delta) > thresh:

		f_beta = np.exp(-X.dot(beta))
		J = -(X.T*f_beta).T  # python trick: make the trailing axes have the same dimension
		delta = np.linalg.solve(J.T.dot(J), J.T.dot(y-f_beta))
		beta += delta

	D = np.array([[beta[0], beta[3], beta[4]], [beta[3], beta[1], beta[5]], [beta[4], beta[5], beta[2]]])

	return D