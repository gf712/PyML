from pyml.maths import Clinear_algebra
from pyml.maths.math_utils import argsort


def dot_product(u, v):
    """
    Dot product using C++ extension.

    :param u:
    :param v:
    :return:

    Example:
    --------

    >>> from pyml.maths.linear_algebra import dot_product
    >>> u = [1, 2, 3]
    >>> v = [4, 5, 6]
    >>> x = dot_product(u, v)
    >>> print(x)
    [32.0]
    >>> A = [[1, -1, 2], [0, -3, 1]]
    >>> x = [2, 1, 0]
    >>> result = dot_product(A, x)
    >>> print(result)
    [1.0, -3.0]
    >>> A = [[0, -4, 4], [-3, -2, 0]]
    >>> B = [[0, 1], [2, 1], [-1, 3]]
    >>> result = dot_product(A, B)
    >>> print(result)
    [[-12.0, 8.0], [-4.0, -5.0]]

    """
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.dot_product(u, v)


def transpose(A):
    """
    Matrix transposition

    :rtype m: list
    :param m: a list of lists representing a matrix A
    :rtype: list
    :return: a list of lists representing the transpose of matrix A

    Example:
    --------

    >>> A = [[0, -4, 4], [-3, -2, 0]]
    >>> print(transpose(A))
    [[0.0, -3.0], [-4.0, -2.0], [4.0, 0.0]]

    """
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.transpose(A)


def broadcast(u, n):
    return [u for i in range(n)]


def subtract(A, B):
    """
    Calculates elementwise difference of each element in a list (vector) or list of lists (matrix)
    If matrix has the same number of columns or rows as the vector the vector is automatically broadcast to fit the matrix

    :type A: list
    :type B: list

    :param A: either a list or a list of lists representing a vector or matrix, respectively
    :param B: either a list or a list of lists representing a vector or matrix, respectively

    :rtype: list
    :return: same format as A (list or list of lists)

    Example:
    --------

    >>> from pyml.maths import subtract
    >>> A = [[0, -4, 4], [-3, -2, 0]]
    >>> B = [0, 1, 2]
    >>> print(subtract(A, B))
    [[0.0, -5.0, 2.0], [-3.0, -3.0, -2.0]]
    >>> C = [0, 1]
    >>> print(subtract(A, C))
    [[0.0, -4.0, 4.0], [-4.0, -3.0, -1.0]]
    >>> D = [[1, 5, 11], [-5, -7, 10]]
    >>> print(subtract(A, D))
    [[-1.0, -9.0, -7.0], [2.0, 5.0, -10.0]]
    """
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.subtract(A, B)


def power(A, n):
    """
    Calculates elementwise power of each element in a list (vector) or list of lists (matrix)

    :type A: list
    :type n: int

    :param A: either a list or a list of lists
    :param n: int to calculate the power

    :rtype: list
    :return: same shape as A (list or list of lists)

    Example:
    --------

    >>> from pyml.maths import power
    >>> A = [[0, -4, 4], [-3, -2, 0]]
    >>> print(power(A, 2))
    [[0.0, 16.0, 16.0], [9.0, 4.0, 0.0]]
    """
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.power(A, n)


def divide(u, n):
    return [x / n for x in u]


def least_squares(X, y):
    """

    :type X: list
    :type y: list

    :param X: list of lists representing a matrix
    :param y: a vector with all targets

    :rtype: list
    :return: list with the same number of dimensions as the number of columns of X with the solution of the least squares problem
    """
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.least_squares(X, y)


def eigen(array, tolerance=1.0e-9, max_iterations=0, sort=True, normalise=True):
    # TODO: write exceptions to help user with errors from the backend
    E, v = Clinear_algebra.eigen_solve(array, tolerance, max_iterations)

    if sort:
        # sort eigenvalues and eigenvectors from biggest to smallest
        idx = argsort(E)[::-1]
        E = [E[i] for i in idx]
        v = [[x[i] for i in idx] for x in v]

    if normalise:
        v = [[x[i] / v[0][i] for i in range(len(x))] for x in v]

    return E, v
