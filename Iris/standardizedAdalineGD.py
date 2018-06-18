import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mlxtend.plotting import plot_decision_regions

from adaline import AdalineGD

dataset = pd.read_csv('iris.csv', header=None)
# print (dataset.tail())

output = dataset.iloc[0:100, 4].values

Y = np.where(output == 'Iris-setosa', -1, 1)
X = dataset.iloc[0:100, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

obj = AdalineGD(epochs=15, learning_rate=0.01)
obj.fit(X_std, Y)
plot_decision_regions(X_std, Y, clf=obj)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(obj.cost_list) + 1), obj.cost_list, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()