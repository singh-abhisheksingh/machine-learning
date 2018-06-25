from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
	def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self, X, Y):
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)
		dim = X_train.shape[1]
		self.indices = tuple(range(dim))
		self.subsets = [self.indices]
		score = self.calculate_score(X_train, Y_train ,X_test, Y_test, self.indices)
		self.scores = [score]

		while dim > self.k_features :
			scores = []
			subsets = []

			for p in combinations(self.indices, r=dim-1):
				score = self.calculate_score(X_train, Y_train ,X_test, Y_test, p)
				scores.append(score)
				subsets.append(p)

			best = np.argmax(scores)
			self.indices = subsets[best]
			self.subsets.append(self.indices)
			dim -=1

			self.scores.append(scores[best])
		self.k_score = self.scores[-1]

		return self

	def transform(self, X):
		return X[:, self.indices]

	def calculate_score(self, X_train, Y_train ,X_test, Y_test, indices):
		self.estimator.fit(X_train[:, indices], Y_train)
		Y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(Y_test, Y_pred)
		return score