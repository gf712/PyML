.. role:: bash(code)
   :language: bash

PyML
====
.. image:: https://coveralls.io/repos/github/gf712/PyML/badge.svg?branch=master
    :target: https://coveralls.io/github/gf712/PyML?branch=master
.. image:: https://travis-ci.org/gf712/PyML.svg?branch=master
    :target: https://travis-ci.org/gf712/PyML

PyML is a Python package with machine learning algorithms written in Python and C/C++.

Installation
============
:bash:`git clone git@github.com:gf712/PyML.git`

:bash:`cd to/folder/of/pyml`

:bash:`python setup.py install`

To verify installation you should run:

:bash:`python setup.py test`

Usage
=====
>>> from pyml.linear_models import LinearRegression
>>> from pyml.datasets import regression
>>> X, y = regression(seed=1970)
>>> lr = LinearRegression(solver='OLS', bias=True)
>>> lr.train(X, y)
>>> lr.coefficients
[0.3011617891659273, 0.9428803588636959]


Changelog
=========
Version 0.2.1 (30/11/2017):
-------------------------
 - Major:
    - PCA implementation
    - Major changes to flatArrays and C++ backend leading to speed improvements
    - Multiclass LogisticRegression

 - Minor:
    - Eigendecomposition of symmetric matrices with Jacobi rotations
    - Improved docs
    - Fixed memory leaks
    - Additional methods for flatArrays (e.g. `diag`)
    - flatArray initialisers

Version 0.2 (16/11/2017):
-------------------------
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

Version 0.1 (09/11/2017):
-------------------------
 - Major:
    - Linear regression (with gradient descent and ordinary least squares)
    - C++ backend

 - Minor:
    - Linear regression with gradient descent about 10x faster with C++ backend
    - Linear regression with OLS (in C++) several orders of magnitude faster than Python gradient descent (depending on set size)
    - KNN uses C++ for distance calculations

Version 0.1dev (02/11/2017):
----------------------------
 - Major:
    - Linear regression with gradient descent
    - KNN, both regressor and classifier
    - Pure Python implementations