from pyml.maths import Clinear_algebra


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
