import numpy as np

class Perceptron(object):

	def __init__(self, learning_rate, epochs):
		self.learning_rate = learning_rate
		self.epochs = epochs

	def fit(self, X, Y):
		self.weight = np.zeros(1 + X.shape[1])
		self.errors = []

		for epoch in range(1, self.epochs):
			error = 0

			for x, y in zip(X, Y):
				weight_update = self.learning_rate * (y - self.predict(x))
				self.weight[1:] += weight_update * x
				self.weight[0] += weight_update
				error += int(weight_update != 0.0)

			self.errors.append(error)

		return self

	def net_input(self, X):
		return np.dot(X, self.weight[1:]) + self.weight[0]

	def predict(self, X):
		return np.where(self.net_input(X) >= 0, 1, -1)