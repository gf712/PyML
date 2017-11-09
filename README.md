PyML
====
[![Coverage Status](https://coveralls.io/repos/github/gf712/PyML/badge.svg)](https://coveralls.io/github/gf712/PyML)
[![Build Status](https://travis-ci.org/gf712/PyML.svg?branch=master)](https://travis-ci.org/gf712/PyML)
PyML is a Python package with machine learning algorithms written in Python and C/C++.

Installation
============
`git clone git@github.com:gf712/PyML.git`

`cd to/folder/of/pyml`

`python setup.py install`

To verify installation you should run:

`python setup.py test`

Usage
=====
```python
>>> from pyml.linear_models import LinearRegression
>>> from pyml.datasets import regression
>>> X, y = regression()
>>> lr = LinearRegression(solver='OLS', bias=True)
>>> lr.train(X, y)
>>> lr.coefficients
[0.3011617891659273, 0.9428803588636959]
```


Changelog
=========
Version 0.1:
 - Major:
    - Linear regression (with gradient descent and ordinary least square)
    - KNN, both regressor and classifier
    - C++ backend
    
 - Minor:
    - Linear regression with gradient descent about x10 faster with C++ backend
    - Linear regression with OLS (in C++) about x10,000 (depending on set) faster than Python gradient descent
    - KNN uses C++ for distance calculations