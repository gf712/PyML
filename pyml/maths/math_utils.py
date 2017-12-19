from collections import Counter
from pyml.maths.CMaths import quick_sort, Cargmax, Cargmin
from pyml.maths.Clinear_algebra import Cmean, Cstd, Cvariance, Ccovariance, Csum


def sort(array, axis=0):
    """
    Sort array elements in ascending order using the quicksort algorithm.

    Args:
        array (list): ist of lists (matrix) or list (vector).
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.

    Returns:
        list: sorted array in ascending order.

    Examples:
        >>> from pyml.maths import sort
        >>> a = [-5, 3, 10, 2, 1, -1]
        >>> print(sort(a))
        [-5.0, -1.0, 1.0, 2.0, 3.0, 10.0]
    """

    return quick_sort(array, axis)[0]


def argsort(array, axis=0):
    """
    Calculate order of elements in array.

    Args:
        array (list): ist of lists (matrix) or list (vector).
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.

    Returns:
        list: indices sorted array in ascending order.

    Examples:
        >>> from pyml.maths import argsort
        >>> a = [-5, 3, 10, 2, 1, -1]
        >>> print(argsort(a))
        [0, 5, 4, 3, 1, 2]
    """
    return quick_sort(array, axis)[1]


def argmin(array, axis=0):
    """
    Returns index of smallest element in a vector, with numpy style column and row wise behaviour for matrices.

    Args:
        array (list): ist of lists (matrix) or list (vector).
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.

    Returns:
        list: vector with argmin for each column/row.

    Examples:
        >>> from pyml.maths import argmin
        >>> a = [-5, 3, 10, 2, 1, -1]
        >>> print(argmin(a))
        [0]
    """
    return Cargmin(array, axis)


def argmax(array, axis=0):
    """
    Returns index of largest element in a vector, with numpy style column and row wise behaviour for matrices.

    Args:
        array (list): ist of lists (matrix) or list (vector).
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.

    Returns:
        list: vector with argmax for each column/row.

    Examples:
        >>> from pyml.maths import argmax
        >>> a = [-5, 3, 10, 2, 1, -1]
        >>> print(argmax(a))
        [2]
    """
    return Cargmax(array, axis)


def max_occurence(array):
    """
    Finds the element with highest occurrence in an array

    Args:
        array (list): a vector

    Returns:
        int: element with highest occurence

    Examples:
        >>> from pyml.maths import max_occurence
        >>> a = [-5, 3, 10, 2, 1, -1, -5, 2, 2]
        >>> print(max_occurence(a))
        2
    """
    count = Counter(array)
    return max(count, key=count.get)


def mean(array, axis=None):
    """
    Numpy style mean of array

    Args:
        array (list): ist of lists (matrix) or list (vector).
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.

    Returns:
        list or int: row/column mean(s) or int of overall mean.

    Examples:
        >>> from pyml.maths import mean
        >>> a = [-5, 3, 10, 2, 5, 0]
        >>> print(mean(a))
        2.5
    """

    return _axis_calc(array, Cmean, axis)


def std(array, degrees_of_freedom=0, axis=None):
    """
    Numpy style standard deviation of array.

    Args:
        array (list): ist of lists (matrix) or list (vector).
        degrees_of_freedom (int): degrees of freedom for standard deviation calculation.
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.

    Returns:
        list or int: row/column standard deviation(s) or int of overall standard deviation.

    Examples:
        >>> from pyml.maths import std
        >>> a = [-5, 3, 8, 2, 0, -1]
        >>> print(std(a))
        3.9756201472921875
    """

    return _axis_calc(array, Cstd, axis, degrees_of_freedom)


def variance(array, degrees_of_freedom=0, axis=None):

    """
    Numpy style variance of array

    Args:
        array (list): ist of lists (matrix) or list (vector).
        degrees_of_freedom (int): degrees of freedom for standard deviation calculation.
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.

    Returns:
        list or int: list with row/column standard deviation(s) or int of overall variance.
    """

    return _axis_calc(array, Cvariance, axis, degrees_of_freedom)


def covariance(array):

    """
    Calculates covariance matrix.

    Args:
        array (list): ist of lists (matrix) or list (vector).

    Returns:
        list: list of lists representing the covariance matrix of array.

    Examples:
        >>> from pyml.datasets import load_iris
        >>> X, y = load_iris()
        >>> [[round(x, 2) for x in i] for i in covariance(X)]
        [[0.68, -0.04, 1.27, 0.51], [-0.04, 0.19, -0.32, -0.12], [1.27, -0.32, 3.09, 1.29], [0.51, -0.12, 1.29, 0.58]]
    """

    return Ccovariance(array)


def sum(A, axis=None):
    """
    Calculates the sum, numpy style
    Args:
        A (list): list of lists (matrix) or list (vector).
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.

    Returns:
        list: sum along given axis

    """
    return _axis_calc(A, Csum, axis)


def _axis_calc(array, func, axis, *args):

    """
    Helper function to perform matrix/vector calculations along a given axis

    Args:
        array (list): list of lists (matrix) or list (vector).
        func (function): calculation to perform.
        axis (int): if array is a matrix this is used to determine whether to order array column or row wise.
        *args: additional arguments to pass to func (e.g. degrees of freedom)

    Returns:
        list: result of func given axis

    Raises:
        TypeError:  raised when array is not a list of lists or list with just ints/floats
        ValueError: raised when list is empty
    """

    if isinstance(array, list):

        if len(array) > 0:
            if isinstance(array[0], list) and isinstance(array[0][0], (float, int)):
                # in this case we have a 2D matrix
                if axis == 1 or axis == 0:
                    return func(array, axis, *args)
                else:
                    return func(func(array, 0), 0)

            elif isinstance(array[0], (int, float)):
                # in this case we have a vector
                return func(array, 0, *args)

            else:
                raise TypeError("Expected a list of lists or a list of int/floats")

        else:
            raise ValueError("Empty list")
    else:
        raise TypeError("Expected a list")
