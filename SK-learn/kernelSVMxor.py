import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

np.random.seed(0)
X_xor = np.random.randn(200, 2)
Y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
Y_xor = np.where(Y_xor, 1, -1)

plt.scatter(X_xor[Y_xor==1, 0], X_xor[Y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[Y_xor==-1, 0], X_xor[Y_xor==-1, 1], c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10)
svm.fit(X_xor, Y_xor)
plot_decision_regions(X_xor, Y_xor, clf=svm)
plt.legend(loc='upper left')
plt.show()