import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df_wine = pd.read_csv('wine.csv', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

df_wine = df_wine[df_wine['Class label'] != 1]
Y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Hue']].values

le = LabelEncoder()
Y = le.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40, random_state=1)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

tree = tree.fit(X_train, Y_train)
Y_train_pred = tree.predict(X_train)
Y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(Y_train, Y_train_pred)
tree_test = accuracy_score(Y_test, Y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

bag = bag.fit(X_train, Y_train)
Y_train_pred = bag.predict(X_train)
Y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(Y_train, Y_train_pred)
bag_test = accuracy_score(Y_test, Y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))