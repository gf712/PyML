from collections import Counter
from .CMaths import quick_sort
from .Clinear_algebra import Cmean


def sort(array):
    """
    sort array elements in ascending order using the quicksort algorithm
    :param array:
    :return: sorted array in ascending order
    """
    return quick_sort(array)[0]


def argsort(array):
    """
    calculate order of elements in array
    :param array:
    :return: sorted indices in ascending order
    """

    return quick_sort(array)[1]


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
    if isinstance(array, list) and len(array) > 0:
        # in this case we have a 2D matrix
        if isinstance(array[0], list) and isinstance(array[0][0], (float, int)):
            dim = len(array[0])
            if axis == 1:
                return [mean([array[x][d] for d in range(dim)]) for x in range(len(array))]
            elif axis == 0:
                return [mean([array[x][d] for x in range(len(array))]) for d in range(dim)]
            else:
                return mean([mean([array[x][d] for d in range(dim)]) for x in range(len(array))])

        elif isinstance(array[0], (int, float)):
            return Cmean(array, 0)

        else:
            raise ValueError("Expected a list of lists or a list of int/floats")

    elif isinstance(array, list):
        raise ValueError("Empty list")

    else:
        raise ValueError("Expected a list")
