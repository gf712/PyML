from pyml.maths import Clinear_algebra
from pyml.maths.math_utils import argsort


def dot_product(u, v):
    """
    Dot product using C++ extension.

    :param u:
    :param v:
    :return:

    >>> u = [1, 2, 3]
    >>> v = [4, 5, 6]
    >>> x = dot_product(u, v)
    >>> print(x)
    32
    >>> A = [[1, -1, 2], [0, -3, 1]]
    >>> x = [2, 1, 0]
    >>> result = dot_product(A, x)
    >>> print(result)
    [1, -3]
    >>> A = [[0, -4, 4], [-3, -2, 0]]
    >>> B = [[0, 1], [2, 1], [-1, 3]]
    >>> result = dot_product(A, B)
    >>> print(result)
    [[0.0, -10.0], [-3.0, -1.0]]

    """
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.dot_product(u, v)


def transpose(m):
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.transpose(m)


def broadcast(u, n):
    return [u for i in range(n)]


def subtract(A, B):
    """
    Calculates elementwise difference of each element in a list (vector) or list of lists (matrix)
    :param A: either a list or a list of lists
    :param B: either a list or a list of lists
    :return: same format as A (list or list of lists)
    """
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.subtract(A, B)


def power(A, n):
    """
    Calculates elementwise power of each element in a list (vector) or list of lists (matrix)
    :param A: either a list or a list of lists
    :param n: int to calculate the power
    :return: same format as A (list or list of lists)
    """
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.power(A, n)


def divide(u, n):
    return [x / n for x in u]


def least_squares(X, y):
    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.least_squares(X, y)


def eigen(array, tolerance=1.0e-9, max_iterations=0, normalise=True):
    # TODO: write exceptions to help user with errors from the backend

    if normalise:
        E, v = Clinear_algebra.eigen_solve(array, tolerance, max_iterations)

        # sort eigenvalues and eigenvectors
        idx = argsort(E)
        E = [E[i] for i in idx]
        v = [[x[i] for i in idx] for x in v]

        v = [[x[i] / v[0][i] for i in range(len(x))] for x in v]

        return E, v

    else:
        return Clinear_algebra.eigen_solve(array, tolerance, max_iterations)
