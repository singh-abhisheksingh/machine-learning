import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

dataset = pd.read_csv('iris.csv', header=None)
# print (dataset.tail())

output = dataset.iloc[0:100, 4].values

Y = np.where(output == 'Iris-setosa', -1, 1)
X = dataset.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length [cm]')
plt.ylabel('sepal length [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(learning_rate=0.1, epochs=10)
ppn.fit(X, Y)
print (ppn.errors)
plt.plot(range(1, len(ppn.errors) + 1), ppn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('No. of misclassifications')
plt.show()