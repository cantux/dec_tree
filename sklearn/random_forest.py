import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree.export import export_text

iris = load_iris()

clf = RandomForestClassifier().fit(iris.data, iris.target)

for e in clf.estimators_:
    r = export_text(e, feature_names=iris['feature_names'])
    print(r)
