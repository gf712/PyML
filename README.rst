.. role:: bash(code)
   :language: bash

PyML
====
.. image:: https://coveralls.io/repos/github/gf712/PyML/badge.svg?branch=master
    :target: https://coveralls.io/github/gf712/PyML?branch=master
.. image:: https://travis-ci.org/gf712/PyML.svg?branch=master
    :target: https://travis-ci.org/gf712/PyML
.. image:: https://anaconda.org/gf712/pyml/badges/installer/conda.svg
    :target: https://conda.anaconda.org/gf712
.. image:: https://anaconda.org/gf712/pyml/badges/downloads.svg
    :target: https://anaconda.org/gf712/pyml
.. image:: https://anaconda.org/gf712/pyml/badges/version.svg
    :target: https://anaconda.org/gf712/pyml

PyML is a Python package with machine learning algorithms written in Python and C/C++.

Installation
============

From source
------------

:bash:`git clone git@github.com:gf712/PyML.git`

:bash:`cd to/folder/of/pyml`

:bash:`python setup.py install`

To verify installation you should run:

:bash:`python setup.py test`

From anaconda
--------------

:bash:`conda install -c gf712 pyml`

Usage
=====
>>> from pyml.linear_models import LinearRegression
>>> from pyml.datasets import regression
>>> X, y = regression(seed=1970)
>>> lr = LinearRegression(solver='OLS', bias=True)
>>> lr.train(X, y)
>>> lr.coefficients
[0.3011617891659273, 0.9428803588636959]