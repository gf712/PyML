from .math_utils import mean
from itertools import starmap
from operator import sub
import linearAlgebraModule


def dot_product(u, v):
    """

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
    [[1], [-3]]

    """

    if len(u) == len(v) and isinstance(u[0], (float, int)) and isinstance(v[0], (float, int)):
        return linearAlgebraModule.dot_product([u], v)
    elif len(u[0]) == len(v):
        return linearAlgebraModule.dot_product(u, v)
    else:
        raise NotImplementedError("This is not the code you are looking for.")


def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def mean_squared_error(y, y_true):
    return mean([(y_i - y_true_i) ** 2 for y_i, y_true_i in zip(y, y_true)])


def mean_absolute_error(y, y_true):
    return mean([abs(y_i - y_true_i) for y_i, y_true_i in zip(y, y_true)])


def broadcast(u, n):
    return [u for i in range(n)]


def subtract(u, v):
    if not isinstance(u, list):
        raise TypeError("Expected a list, but got {} instead.".format(type(u)))
    if not isinstance(v, list):
        raise TypeError("Expected a list, but got {} instead.".format(type(v)))
    return linearAlgebraModule.subtract(u, v)


def power(u, n):
    return linearAlgebraModule.power(u, n)


def divide(u, n):
    return [x / n for x in u]
