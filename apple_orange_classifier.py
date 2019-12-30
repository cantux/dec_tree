#!/usr/bin/env python

from sklearn import tree

def train_classifier(features, labels):
   # Training classifier
    classifier = tree.DecisionTreeClassifier() # using decision tree classifier
    classifier = classifier.fit(features, labels) # Find patterns in data
    return classifier

def test():
    # Gathering training data
    # features = [[155, “rough”], [180, “rough”], [135, “smooth”], [110, “smooth”]] # Input to classifier
    features = [[155, 0], [180, 0], [135, 1], [110, 1]] # scikit-learn requires real-valued features
    # labels = [“orange”, “orange”, “apple”, “apple”] # output values
    labels = [1, 1, 0, 0]
 
    dec_tree_classifier = train_classifier(features, labels)
    assert dec_tree_classifier.predict([[120, 1]]) == 0

if __name__ == "__main__":
    test()

