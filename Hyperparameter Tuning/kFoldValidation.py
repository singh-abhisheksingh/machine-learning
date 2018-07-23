import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv('breastcancer.data', header=None)
X = df.loc[:, 2:].values
Y = df.loc[:, 1].values
le = LabelEncoder()
Y = le.fit_transform(Y)

print(le.transform(['M', 'B']))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])

kfold = StratifiedKFold(n_splits=10, random_state=1)
scores = []
for k, (train, test) in enumerate(kfold.split(X_train, Y_train)):
	pipe_lr.fit(X_train[train], Y_train[train])
	score = pipe_lr.score(X_train[test], Y_train[test])
	scores.append(score)
	print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(Y_train[train]), score))

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


scores = cross_val_score(estimator=pipe_lr, X=X_train, y=Y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))