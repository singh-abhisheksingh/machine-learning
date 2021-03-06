import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons
from matplotlib.ticker import FormatStrFormatter

from kernelRBF import rbf_kernel_pca

X, Y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[Y==0, 0], X[Y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[Y==1, 0], X[Y==1, 1], color='blue', marker='o', alpha=0.5)
plt.show()

X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(9,4))
ax[0].scatter(X_kpca[Y==0, 0], X_kpca[Y==0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[Y==1, 0], X_kpca[Y==1, 1], color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[Y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[Y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()