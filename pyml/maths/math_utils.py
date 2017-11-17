from collections import Counter
from .CMaths import quick_sort
from .Clinear_algebra import Cmean, Cstd, Cvariance, Ccovariance
from math import exp


def sort(array, axis=0):
    """
    sort array elements in ascending order using the quicksort algorithm
    :param array:
    :return: sorted array in ascending order
    """
    return quick_sort(array, axis)[0]


def argsort(array, axis=0):
    """
    calculate order of elements in array
    :param array:
    :return: sorted indices in ascending order
    """
    return quick_sort(array, axis)[1]


def max_occurence(array):
    """
    find the element with highest occurrence in an array
    :param array:
    :return:
    """
    count = Counter(array)
    return max(count, key=count.get)


def mean(array, axis=None):
    """
    numpy style mean of array
    :param array:
    :param axis:
    :return:
    """
    if isinstance(array, list):

        if len(array) > 0:
            if isinstance(array[0], list) and isinstance(array[0][0], (float, int)):
                # in this case we have a 2D matrix
                if axis == 1 or axis == 0:
                    return Cmean(array, axis)
                else:
                    # return mean([mean([array[x][d] for d in range(dim)]) for x in range(len(array))])
                    return Cmean(Cmean(array, 0), 0)

            elif isinstance(array[0], (int, float)):
                # in this case we have a vector
                return Cmean(array, 0)

            else:
                raise TypeError("Expected a list of lists or a list of int/floats")

        else:
            raise ValueError("Empty list")
    else:
        raise TypeError("Expected a list")


def std(array, degrees_of_freedom=0, axis=None):

    """
    numpy style standard deviation of array
    :param array:
    :param axis:
    :return:
    """
    if isinstance(array, list):

        if len(array) > 0:
            if isinstance(array[0], list) and isinstance(array[0][0], (float, int)):
                # in this case we have a 2D matrix
                if axis == 1 or axis == 0:
                    return Cstd(array, degrees_of_freedom, axis)
                else:
                    raise NotImplementedError("This is not the code you are looking for.")

            elif isinstance(array[0], (int, float)):
                # in this case we have a vector
                return Cstd(array, degrees_of_freedom, 0)

            else:
                raise TypeError("Expected a list of lists or a list of int/floats")

        else:
            raise ValueError("Empty list")
    else:
        raise TypeError("Expected a list")


def variance(array, degrees_of_freedom=0, axis=None):

    """
    numpy style standard deviation of array
    :param array:
    :param axis:
    :return:
    """
    if isinstance(array, list):

        if len(array) > 0:
            if isinstance(array[0], list) and isinstance(array[0][0], (float, int)):
                # in this case we have a 2D matrix
                if axis == 1 or axis == 0:
                    return Cvariance(array, degrees_of_freedom, axis)
                else:
                    raise NotImplementedError("This is not the code you are looking for.")

            elif isinstance(array[0], (int, float)):
                # in this case we have a vector
                return Cvariance(array, degrees_of_freedom, 0)

            else:
                raise TypeError("Expected a list of lists or a list of int/floats")

        else:
            raise ValueError("Empty list")
    else:
        raise TypeError("Expected a list")


def covariance(array):

    """

    :param array: list of lists
    :return:
    """

    return Ccovariance(array)


def sigmoid(array):
    """
    Python implementation of element wise sigmoid
    :param array: list
    :return: list
    """
    return [1 / (1 + exp(-array_i)) for array_i in array]
