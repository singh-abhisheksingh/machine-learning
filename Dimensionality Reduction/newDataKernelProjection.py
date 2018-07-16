import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons

from kernelRBFupdated import rbf_kernel_pca

X, Y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[Y==0, 0], X[Y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[Y==1, 0], X[Y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)

x_new = X[25]
x_proj = alphas[25]

def project_x(x_new, X, gamma, alphas, lambdas):
	pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
	k = np.exp(-gamma * pair_dist)
	return k.dot(alphas / lambdas)

x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)

plt.scatter(alphas[Y==0, 0], np.zeros((50)), color='red', marker='^',alpha=0.5)
plt.scatter(alphas[Y==1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()