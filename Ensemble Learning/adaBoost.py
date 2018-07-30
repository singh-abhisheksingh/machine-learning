import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
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

tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=0)

tree = tree.fit(X_train, Y_train)
Y_train_pred = tree.predict(X_train)
Y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(Y_train, Y_train_pred)
tree_test = accuracy_score(Y_test, Y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

ada = ada.fit(X_train, Y_train)
Y_train_pred = ada.predict(X_train)
Y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(Y_train, Y_train_pred)
ada_test = accuracy_score(Y_test, Y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))