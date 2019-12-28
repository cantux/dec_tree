# dec_tree
decision trees with sklearn

## environment

envrionment is managed by conda.

### install conda
https://docs.conda.io/en/latest/miniconda.html and grab version that suits you.

### create a new conda environment


## terms

### Entropy:

Measure of randomness or unpredictability in the dataset.

we want to drive towards a lower entropy

ex: given a set elements belonging to two classes, we want to find decisions that will segregate the data.

high -> low

### Information gain:

Measure of decrease in entropy after the dataset is split.

### Impurity

If a certain feature(or group of features) can't fully classify data, they are called impure.

Take the example:

|                  | Chest Pain | No Chest Pain |
|------------------|------------|---------------|
| Heart Disease    | 105        | 34            |
| No Heart Disease | 39         | 125           |

Posterior, probability having heart disease given that patient has chest pain ->

P(H|CP) = 
