from pyml.metrics.CMetrics import norm


def calculate_distance(u, v, p):
    """
    Calculates the distance between two vectors given a norm p.

    Args:
        u (list): vector.
        v (list): vector.
        p (int or str): norm to calculate distance.

    Returns:
        float: distance between vectors u and v.

    Examples:
        >>> from pyml.metrics.distances import calculate_distance
        >>> a = [-5, 3, 8, 2, 0, -1]
        >>> b = [-10, 2, -5, 6, 2, 0]
        >>> print(calculate_distance(a, b, 'l1'))
        26.0
        >>> print(calculate_distance(a, b, 'l2'))
        14.696938456699069
    """

    if isinstance(p, str):
        if p == 'l1':
            p = 1
        elif p == 'l2':
            p = 2
        else:
            raise ValueError("Unknown norm.")

    if p == 1:
        return manhattan_distance(u, v)
    elif p == 2:
        return euclidean_distance(u, v)
    else:
        return norm(u, v, p)


def euclidean_distance(u, v):
    """
    Calculates the euclidean distance between two vectors.

    Args:
        u (list): vector.
        v (list): vector.

    Returns:
        float: euclidean distance between vectors u and v.

    Examples:
        >>> from pyml.metrics.distances import euclidean_distance
        >>> a = [-5, 3, 8, 2, 0, -1]
        >>> b = [-10, 2, -5, 6, 2, 0]
        >>> print(euclidean_distance(a, b))
        14.696938456699069
    """

    return norm(u, v, 2)


def manhattan_distance(u, v):
    """
    Calculates the Manhattan distance between two vectors.

    Args:
        u (list): vector.
        v (list): vector.

    Returns:
        float: Manhattan distance between vectors u and v.

    Examples:
        >>> from pyml.metrics.distances import manhattan_distance
        >>> a = [-5, 3, 8, 2, 0, -1]
        >>> b = [-10, 2, -5, 6, 2, 0]
        >>> print(manhattan_distance(a, b))
        26.0
    """

    return norm(u, v, 1)
