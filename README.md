PyML
====

[![Anaconda-Server Badge](https://anaconda.org/gf712/pyml/badges/installer/conda.svg)](https://conda.anaconda.org/gf712)
[![Anaconda-Server Badge](https://anaconda.org/gf712/pyml/badges/downloads.svg)](https://anaconda.org/gf712/pyml)
[![Anaconda-Server Badge](https://anaconda.org/gf712/pyml/badges/version.svg)](https://anaconda.org/gf712/pyml)

Branch      | Linux/OSX | Coverage | Maintainability |
------------|-----------|----------|-----------| 
[master](https://github.com/boostorg/beast/tree/master)   | [![Build Status](https://travis-ci.org/gf712/PyML.svg?branch=master)](https://travis-ci.org/gf712/PyML) | [![Coverage Status](https://coveralls.io/repos/github/gf712/PyML/badge.svg)](https://coveralls.io/github/gf712/PyML) | [![Maintainability](https://api.codeclimate.com/v1/badges/0741a85a993857bfa5f7/maintainability)](https://codeclimate.com/github/gf712/PyML/maintainability)
[dev](https://github.com/boostorg/beast/tree/dev)  |  [![Build Status](https://travis-ci.org/gf712/PyML.svg?branch=dev)](https://travis-ci.org/gf712/PyML) | [![Coverage Status](https://coveralls.io/repos/github/gf712/PyML/badge.svg?branch=dev)](https://coveralls.io/github/gf712/PyML?branch=dev) | [![Maintainability](https://api.codeclimate.com/v1/badges/0741a85a993857bfa5f7/maintainability)](https://codeclimate.com/github/gf712/PyML/dev/maintainability)

PyML is a Python package with machine learning algorithms written in Python and C/C++.
 
PyML is currently supported on Linux and OSX, with GCC>=4.8 and Clang>=3.3, on both platforms.

Installation
============

## From source

`git clone git@github.com:gf712/PyML.git`

`cd to/folder/of/pyml`

`python setup.py install`

To verify installation you should run:

`python setup.py test`

## From anaconda

`conda install -c gf712 pyml`

Usage
=====
```python
>>> from pyml.linear_models import LinearRegression
>>> from pyml.datasets import regression
>>> X, y = regression(seed=1970)
>>> lr = LinearRegression(solver='OLS', bias=True)
>>> lr.train(X, y)
>>> lr.coefficients
[0.3011617891659273, 0.9428803588636959]
```

Changelog
=========
## Version 0.2.1.1 (TBC):
 - Major:
    - Library now available on anaconda
 
 - Minor:
    - Backend speed improvements
    - PEP8 formatting
    - Code maintainability tracking with codeclimate

## Version 0.2.1 (05/12/2017):
 - Major:
    - Machine learning algorithms:
        - PCA implementation
        - Multiclass LogisticRegression
    - Optimisation algorithms:
        - Mini batch gradient descent
        - Gradient descent with momentum
        - Nesterov optimisation
        - Adagrad optimisation
        - Adadelta optimisation 

    - Backend:
        - Major changes to flatArrays and C++ backend leading to speed improvements
    
 - Minor:
    - Eigendecomposition of symmetric matrices with Jacobi rotations
    - Improved docs
    - Fixed memory leaks
    - increased coverage
    
## Version 0.2 (16/11/2017):
 - Major:
    - KMeans implementation
    - Logistic regression implementation
    - New C++ class to represent 2D arrays more efficiently

 - Minor:
    - Cleaned up C++ code
    - Norm is calculated with pure C++ (Python only used to provide `euclidean_distance` and `manhattan_distance` interface)
    - Faster implementation of quick sort algorithm (with C++)
    - Fixed `argsort` behaviour
    - Overall speed improvements using flat arrays (e.g. matrix matrix multiplication ~2 times faster)
    - More tests and increased code coverage
    - Started fixing memory leaks
    - column and row wise `sort` and `argsort`

## Version 0.1 (09/11/2017):
 - Major:
    - Linear regression with ordinary least squares
    - C++ backend
    
 - Minor:
    - Linear regression with gradient descent about 10x faster with C++ backend
    - Linear regression with OLS (in C++) several orders of magnitude faster than Python gradient descent (depending on set size)
    - KNN uses C++ for distance calculations
    
## Version 0.1dev (02/11/2017):
 - Major:
    - Linear regression with gradient descent 
    - KNN, both regressor and classifier
    - Pure Python implementations