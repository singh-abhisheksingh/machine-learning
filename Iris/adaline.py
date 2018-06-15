import numpy as np

class AdalineGD(object):

	def __init__(self, learning_rate=0.01, epochs=50):
		self.learning_rate = learning_rate
		self.epochs = epochs

	def fit(self, X, Y):
		self.weight = np.zeros(1 + X.shape[1])
		self.cost_list = []

		for i in range(self.epochs):
			outputs = self.net_input(X)
			errors = (Y - outputs)
			self.weight[1:] += self.learning_rate * X.T.dot(errors)
			self.weight[0] += self.learning_rate * errors.sum()
			cost = (errors ** 2).sum() / 2.0
			self.cost_list.append(cost)

		return self

	def net_input(self, X):
		return np.dot(X, self.weight[1:]) + self.weight[0]

	def activation(self, X):
		return self.net_input(X)

	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, 1, -1)