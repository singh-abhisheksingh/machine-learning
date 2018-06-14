import numpy as np
from matplotlib import pyplot as plt

dataset = np.array([
	[-2, 4, -1],
	[4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
	])
output = np.array([-1, -1, 1, 1, 1])

for count, point in enumerate(dataset):
	print (count, "\t", point)

	if (count < 2):
		plt.scatter(point[0], point[1], s=120, marker="_", linewidths=2)
	else:
		plt.scatter(point[0], point[1], s=120, marker="+", linewidths=2)
plt.show()

def svm_gradient(X, Y):
	weight = np.zeros(len(X[0]))
	print ("Initial Weight:",weight)
	learning_rate = 1
	epochs = 100000
	errors = []

	for epoch in range(1,epochs):
		error = 0

		for i, point in enumerate(X):
			if ( Y[i] * np.dot(X[i],weight)) < 1:
				weight = weight + learning_rate * ((X[i] * Y[i]) + (-2 * (1/epoch) * weight))
				error = 1
			else:
				weight = weight + learning_rate * (-2 * (1/epoch) * weight)

		errors.append(error)

	print ("Final Weight:",weight)

	plt.axes().set_yticklabels([])
	plt.ylim(0.5, 1.5)
	plt.plot(errors, '|')
	plt.xlabel('Epochs')
	plt.ylabel('Misclassified')
	plt.show()

	return weight

wt = svm_gradient(dataset, output)