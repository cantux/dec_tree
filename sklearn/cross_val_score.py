import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
# print("iris data: " + str(X))
y = iris.target

clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)

# 5-Fold by default
scores = cross_val_score(clf, X, y)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
