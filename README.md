# dec_tree
decision trees with sklearn

## environment

envrionment is managed by conda.

### install conda
https://docs.conda.io/en/latest/miniconda.html and grab version that suits you.

### create a new conda environment


## Definitions

### Entropy:

Measure of randomness or unpredictability in the dataset.

Drive towards a lower entropy

ex: given a set elements belonging to two classes, we want to find decisions that will segregate the data.

high -> low

### Impurity

If a certain feature(or group of features) can't fully classify data, they are called impure.

## Attribute selection measures

### Information gain:

Measure of decrease in entropy after the dataset is split.

Gini index mesaured by:

G = sum over classses K( p_m_k * (1 - p_m_k)) -> Gini index takes a small value if all of the pmk's are close to zero or one.

Entropy is measured by:

Entropy before split H(Y) = D = - sum over classes k( p_m_k * log(p_m_k))

Entropy of Y after split:

H(Y|X) = - sum over k(P(X = x_j) * sum over m( P(Y=y_i|X=x_j) * log(P(Y=y_i|X=x_j))

Information gain: IG(X) = H(X) - H(Y|X)

### Algorithm to improve purity

1- Try all combination of x1, x2 ... / value1a, value1b, value2a, value2b ..
2- Pick a combination such that the divided data set has a better combined purity
3- To avoid the decition tree overfits the training data, divide only when the purity after divide exceed a threshold value.
This is called pre-pruining. Andrew Moore uses magic threshold 0.05

* Post pruning is less prone to overfitting

#### Resources

http://www.cs.cmu.edu/~tom/10601_fall2012/lectures.shtml
http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf -> chapter 8.1.2
## Regression vs Classification

Regression trees yield a real value while classification trees yield a category.


## Errors on prediction models

There are two main subcomponents: bias and variance. Understanding these errors help us diagnose model results and avoid the mistake of over and under-fitting

http://scott.fortmann-roe.com/docs/BiasVariance.html

## Bias and Variance

Bias is defined in three ways:

- conceptually
- graphically
- mathematically

### Conceptual Definition

*Error due to Bias:* Difference between expected(or average) and the correct value which we are trying to predict.
If we ran model building multiple times with different bags of data we will have different validation results. 
Bias measures how far off in general these models' oredictions are from the correct value


*Error due to Variance:* Taken as the variability of a model prediction for a given data point.
If we ran model building multiple times, variance is how much predictions for a given point vary between different realizations of the model.
High variance overfitting

### Graphical Definition

Bullseye diagram

### Mathematical Definition

variable we are trying to predict Y

covariates as X

Y = f(X) + e where e is normally distributed with a mean of zero

e ~ N(0, s_e)

We may estimate a mode f'(X) of f(X)

Err(X) = E[(Y - f'(x))^2]

Then decomposed in to bias and variance components:

Err(x) = ( E(f'(X)) - f(x) )^2 + E[f'(x)- E[f'(X)]^2] + s_e^2

Err(x) = Bias^2 + Variance + Irreducable Error


Bagging and aggregating fights variance and bias. Random forests(ensambles) use this method.

Sweet spot for any model is the level of complexity at which the increase in bias is equivalent to the reduction in variance.

delta_bias/delta_complexity = - delta_variance / delta_complexity  -- take derivatives for regression.





