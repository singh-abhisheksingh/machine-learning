import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from adaline import AdalineGD

dataset = pd.read_csv('iris.csv', header=None)
# print (dataset.tail())

output = dataset.iloc[0:100, 4].values

Y = np.where(output == 'Iris-setosa', -1, 1)
X = dataset.iloc[0:100, [0, 2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

obj1 = AdalineGD(epochs=10, learning_rate=0.01).fit(X, Y)
ax[0].plot(range(1, len(obj1.cost_list) + 1), np.log10(obj1.cost_list), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-errors)')
ax[0].set_title('Adaline - Learning rate 0.01')

obj2 = AdalineGD(epochs=10, learning_rate=0.0001).fit(X, Y)
ax[1].plot(range(1, len(obj2.cost_list) + 1), obj2.cost_list, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-errors')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()