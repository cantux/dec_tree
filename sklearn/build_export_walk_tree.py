import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
# print("iris data: " + str(X))
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)

# DecisionTreeClassifier structure:
# children_left/right, feature ids, thresholds are all arrays of data indexed by node ids
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
value = estimator.tree_.value

def mosys_predict(node_id, data_point_dct):
    if children_left[node_id] == children_right[node_id]:
# this is the leaf return category
        return np.argmax(value[node_id])
    else:
        # this is a decision node recurse
        feature_id = feature[node_id]
        if data_point_dct[feature_id] <= threshold[node_id]:
            return mosys_predict(children_left[node_id], data_point_dct)
        else:
            return mosys_predict(children_right[node_id], data_point_dct)

# print("x_test: " + str( X_test))
# print("y_test: " + str(y_test))
for x in X_test:
    print("x: " + str(x))
    assert mosys_predict(0, x) == estimator.predict(x.reshape(1,-1))[0]
