import numpy as np
import matplotlib.pyplot as plt

from skPerceptron import plot_decision_regions

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train_std, Y_train)

Y_prediction = tree.predict(X_test_std)
print ("Misclassified samples: %d" % (Y_test != Y_prediction).sum())
print ("Accuracy: %.2f" % accuracy_score(Y_test, Y_prediction))

X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((Y_train, Y_test))
plot_decision_regions(X_combined_std, Y=Y_combined, classifier=tree, test_idx=(range(105, 150)))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])