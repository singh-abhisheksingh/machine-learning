from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
	sq_dists = pdist(X, 'sqeuclidean')
	mat_sq_dists = squareform(sq_dists)
	K = exp(-gamma * mat_sq_dists)
	
	N = K.shape[0]
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	eigvals, eigvecs = eigh(K)
	alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
	lambdas = [eigvals[-i] for i in range(1,n_components+1)]
	
	return alphas, lambdas