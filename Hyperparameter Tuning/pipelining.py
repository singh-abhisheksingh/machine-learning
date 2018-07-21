import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv('breastcancer.data', header=None)
X = df.loc[:, 2:].values
Y = df.loc[:, 1].values
le = LabelEncoder()
Y = le.fit_transform(Y)

print(le.transform(['M', 'B']))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, Y_train)

print('Test Accuracy: %0.3f' % pipe_lr.score(X_test, Y_test))