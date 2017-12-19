from pyml.maths import Clinear_algebra
from pyml.maths.math_utils import argsort


def dot_product(u, v):
    """
    Matrix/matrix, matrix/vector and vector/vector dot product.

    Args:
        u (list): either a list or a list of lists representing a vector or matrix, respectively.
        v (list): either a list or a list of lists representing a vector or matrix, respectively.

    Returns:
        list:

    Examples:
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
    Matrix transposition.

    Args:
        A (list): a list of lists representing a matrix.

    Returns:
        list: a list of lists representing the transpose of matrix A.

    Examples:

    >>> A = [[0, -4, 4], [-3, -2, 0]]
    >>> print(transpose(A))
    [[0.0, -3.0], [-4.0, -2.0], [4.0, 0.0]]
    """

    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.transpose(A)


def add(A, B):
    """
    Calculates elementwise addition of each element in a list (vector) or list of lists (matrix).
    If matrix has the same number of columns or rows as the vector the vector is automatically broadcast to fit the matrix.

    Args:
        A (list or scalar):  either a list or a list of lists representing a vector or matrix, respectively.
        B (list):  either a list or a list of lists representing a vector or matrix, respectively.

    Returns:
        list: same format as A (list or list of lists).

    Raises:
        DimensionMismatchException: if A and B do not have matching dimensions

    Examples:
        >>> from pyml.maths import add
        >>> A = [[0, -4, 4], [-3, -2, 0]]
        >>> B = [0, 1, 2]
        >>> print(add(A, B))
        [[0.0, -3.0, 6.0], [-3.0, -1.0, 2.0]]
        >>> C = [0, 1]
        >>> print(add(A, C))
        [[0.0, -4.0, 4.0], [-2.0, -1.0, 1.0]]
        >>> D = [[1, 5, 11], [-5, -7, 10]]
        >>> print(add(A, D))
        [[1.0, 1.0, 15.0], [-8.0, -9.0, 10.0]]
    """

    return _template_func(A, B, Clinear_algebra.add)


def subtract(A, B):

    """
    Calculates elementwise difference of each element in a list (vector) or list of lists (matrix).
    If matrix has the same number of columns or rows as the vector the vector is automatically broadcast to fit the matrix.
    
    Args:
        A (list or scalar): either a list or a list of lists representing a vector or matrix, respectively.
        B (list): either a list or a list of lists representing a vector or matrix, respectively.

    Returns:
        list: same format as A (list or list of lists).

    Raises:
        DimensionMismatchException: if A and B do not have matching dimensions

    Examples:
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

    return _template_func(A, B, Clinear_algebra.subtract)


def power(A, n):
    """
    Calculates elementwise power of a list (vector) or list of lists (matrix).

    Args:
        A (list): either a list or a list of lists.
        n (float): exponent.

    Returns:
        list: same shape as A (list or list of lists).

    Examples:
        >>> from pyml.maths import power
        >>> A = [[0, -4, 4], [-3, -2, 0]]
        >>> print(power(A, 2))
        [[0.0, 16.0, 16.0], [9.0, 4.0, 0.0]]
    """

    # TODO: write exceptions to help user with errors from the backend
    return Clinear_algebra.power(A, n)


def multiply(A, B):
    """
    Calculates elementwise multiplication of a list (vector) or list of lists (matrix) with another vector or matrix
    and automatic broadcasting if needed.

    Args:
        A (list): either a list (vector) or a list of lists (matrix).
        B (list): either a list (vector) or a list of lists (matrix) or a constant.

    Returns:
        list: matrix divided by constant n with the same shape as A (list or list of lists).

    Examples:
        >>> from pyml.maths import multiply
        >>> A = [[0, -4, 4], [-3, -2, 0]]
        >>> print(multiply(A, 2))
        [[0.0, -8.0, 8.0], [-6.0, -4.0, 0.0]]
    """

    return _template_func(A, B, Clinear_algebra.multiply)


def divide(A, B):

    """
    Calculates elementwise division of a list (vector) or list of lists (matrix).
    Args:
        A (list): either a list or a list of lists.
        B (list or float): scalar to perform division.

    Returns:
        list: matrix divided by constant n with the same shape as A (list or list of lists).

    Raises:
        ZeroDivisionError: If any of the values in B is 0.

    Examples:
        >>> from pyml.maths import divide
        >>> A = [[0, -4, 4], [-3, -2, 0]]
        >>> print(divide(A, 2))
        [[0.0, -2.0, 2.0], [-1.5, -1.0, 0.0]]
    """

    try:
        return _template_func(A, B, Clinear_algebra.divide)
    except Clinear_algebra.ZeroDivisionError as e:
        # raises Python builtin error instead
        raise ZeroDivisionError(e)


def determinant(A):
    """
    Calculates the determinant of matrix A.

    Args:
        A (list): list of lists representing a square matrix.

    Returns:
        float: determinant of matrix A.

    Raises:
        LinearAlgebraException: if A is not a square matrix.

    Examples:
        >>> from pyml.maths import determinant
        >>> A = [[3, 1], [5, 2]]
        >>> print(determinant(A))
        1.0
    """

    return Clinear_algebra.determinant(A)


def least_squares(X, y):
    """
    Solves a system of linear equations using Gaussian elimination.

    Args:
        X (list): list of lists representing a matrix.
        y (list): a vector with all targets.

    Returns:
        list: list with the same number of dimensions as the number of columns of X with the solution of the system of
            linear equations.

    """

    return Clinear_algebra.least_squares(X, y)


def eigen(array, tolerance=1.0e-9, max_iterations=0, sort=True, normalise=True):
    """
    Eigendecomposition of square matrices with Jacobi rotations.

    Args:
        array (list): list of lists representing a matrix.
        tolerance (Optional[float]): early stopping parameter of Jacobi matrix decomposition algorithm.
            Defaults to 1.0e-9.
        max_iterations (Optional[int]): maximum number of iterations of Jacobi matrix decomposition algorithm.
            Defaults to 0, and in the C++ code is calculated as 5 * n ** 2.
        sort (Optional[bool]): whether or not to sort eigenvalues (descending) and respective eigenvectors.
        normalise (Optional[bool]): whether or not to normalise eigenvectors using eigenvectors of the first eigenvalue.

    Returns:
        tuple: (eigenvalues (list), eigenvectors(list of lists)).

    Examples:
        >>> from pyml.maths import eigen
        >>> S = [[3., -1, 0], [-1, 2, -1], [0, -1, 3]]
        >>> w, v = eigen(S)
        >>> v = [[round(x, 1) for x in row] for row in v]
        >>> w = [round(x, 1) for x in w]
        >>> print(v)
        [[1.0, 1.0, 1.0], [-1.0, -0.0, 2.0], [1.0, -1.0, 1.0]]
        >>> print(w)
        [4.0, 3.0, 1.0]

    """

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


def _template_func(A, B, func):
    """
    Helper function that takes function and applies to A and B, and changes B to list if it's a scalar
    Args:
        A (list):
        B (list or float or int):
        func (function):

    Returns:
        list: function result with params A and B
    """

    if isinstance(B, (float, int)):
        B = [B]

    return func(A, B)
