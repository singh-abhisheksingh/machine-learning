from numpy.random import seed
import numpy as np

class AdalineSGD(object):

	def __init__(self, learning_rate=0.01, epochs=10, shuffle=True, random_state=None):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.weight_initialized = False
		self.shuffle = shuffle

		if random_state:
			seed(random_state)

	def fit(self, X, Y):
		self.initialize_weights(X.shape[1])
		self.cost_list = []

		for i in range(self.epochs):
			if self.shuffle:
				X, Y = self.shuffle_function(X, Y)
			cost = []

			for x, y in zip(X, Y):
				cost.append(self.update_weights(x, y))

			avg_cost = sum(cost)/len(Y)
			self.cost_list.append(avg_cost)

		return self

	def partial_fit(self, X, Y):
		if not self.weight_initialized:
			self.initialize_weights(X.shape[1])
		if Y.ravel().shape[0] > 1:
			for x, y in zip(X, Y):
				self.update_weights(x, y)
		else:
			self.update_weights(X, Y)

		return self

	def shuffle_function(self, X, Y):
		r = np.random.permutation(len(Y))
		return X[r], Y[r]

	def initialize_weights(self, m):
		self.weight = np.zeros(1 + m)
		self.weight_initialized = True

	def update_weights(self, x, y):
		output = self.net_input(x)
		error = (y - output)
		self.weight[1:] += self.learning_rate * x.dot(error)
		self.weight[0] += self.learning_rate * error
		cost = 0.5 * error ** 2
		return cost

	def net_input(self, X):
		return np.dot(X, self.weight[1:]) + self.weight[0]

	def activation(self, X):
		return self.net_input(X)

	def predict(self, X):
		return np.where(self.activation(X) >= 0.0, 1, -1)