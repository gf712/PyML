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
    if isinstance(u[0], list):
        if isinstance(v[0], list):
            # matrix matrix multiplication
            return matrix_product(u, v)
        elif isinstance(v[0], (float, int)):
            # matrix vector product
            return Clinear_algebra.dot_product(u, v)
        else:
            raise NotImplementedError("This is not the code you are looking for.")
    elif isinstance(u[0], (float, int)) and isinstance(v[0], (float, int)):
        # vector vector product
        return Clinear_algebra.dot_product(u, v)
    else:
        raise NotImplementedError("This is not the code you are looking for.")


def transpose(m):
    return Clinear_algebra.transpose(m)


def broadcast(u, n):
    return [u for i in range(n)]


def subtract(u, v):
    if not isinstance(u, list):
        raise TypeError("Expected a list, but got {} instead.".format(type(u)))
    if not isinstance(v, list):
        raise TypeError("Expected a list, but got {} instead.".format(type(v)))
    return Clinear_algebra.subtract(u, v)


def power(A, n):
    """
    Calculates elementwise power of each element in a list (vector) or list of lists (matrix)
    :param A: either a list or a list of lists
    :param n: int to calculate the power
    :return: same format as A (list or list of lists)
    """
    return Clinear_algebra.power(A, n)


def divide(u, n):
    return [x / n for x in u]


def matrix_product(m, n):
    return Clinear_algebra.matrix_product(m, n)


def least_squares(X, y):
    return Clinear_algebra.least_squares(X, y)
