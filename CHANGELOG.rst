Changelog
=========
Version 0.2.1.1 (14/12/2017):
-----------------------------
 - Major:
    - Library now available on anaconda

 - Minor:
    - Backend speed improvements

Version 0.2.1 (05/12/2017):
---------------------------
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