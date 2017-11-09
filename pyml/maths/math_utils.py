import copy
from collections import Counter


def sort(array):
    """
    sort array elements in ascending order using the quicksort algorithm
    :param array:
    :return: a sorted array
    """
    # create a copy of the array so that we don't change the original
    sorted_array = copy.copy(array)
    quicksort(sorted_array, 0, len(sorted_array) - 1)
    return sorted_array


def argsort(array):
    """
    calculate order of elements in array
    :param array:
    :return:
    """

    sorted_array = sort(array)

    return [sorted_array.index(i) for i in array]


def quicksort(array, low, high):
    """
    quicksort helper function
    :param array:
    :param low:
    :param high:
    :return:
    """
    if low < high:
        p = partition(array, low, high)
        quicksort(array, low, p - 1)
        quicksort(array, p + 1, high)


def partition(array, low, high):
    """
    partition the array (see quicksort implementation)
    :param array:
    :param low:
    :param high:
    :return:
    """
    pivot = array[low]
    left = low + 1
    right = high

    while True:

        while left <= right and array[left] <= pivot:
            left += 1

        while array[right] >= pivot and right >= left:
            right -= 1

        if right < left:
            break
        else:
            swap(array, left, right)

    swap(array, low, right)

    return right


def swap(array, i, j):
    """
    swap two elements of an array
    :param array:
    :param i:
    :param j:
    :return:
    """
    old = array[i]
    array[i] = array[j]
    array[j] = old


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
    mean of array
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
            return sum(array) / len(array)

        else:
            raise ValueError("Expected a list of lists or a list of int/floats")

    elif isinstance(array, list):
        raise ValueError("Empty list")

    else:
        raise ValueError("Expected a list")
