
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.tree.export import export_text

iris = load_iris()

# plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
# plot_tree(clf, filled=True)
# plt.show()

r = export_text(clf, feature_names=iris['feature_names'])
print(r)
