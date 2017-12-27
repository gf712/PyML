import math


def sigmoid(u):

    """
    Python implementation of element wise sigmoid of a vector.

    Args:
        u (list): list representing a vector.

    Returns:
        list: sigmoid of u

    Raises:
        TypeError: raised if list passed is not a list of scalars.
    """

    if isinstance(u[0], (float, int)) and len(u) > 0:
        return [1 / (1 + math.exp(-el)) for el in u]

    else:
        raise TypeError("Expected a list of scalars.")


def softmax(u):

    """
    Computes softmax of a matrix/vector u.

    Args:
        u (list): vector.

    Returns:
        list: softmax of vector u.

    Raises:
        TypeError: raised if argument is not a list of scalars (vector),
                   or a list of lists (matrix).
    """

    if isinstance(u[0], list):
        # matrix
        return [softmax(row) for row in u]

    elif isinstance(u[0], (float, int)):
        # vector
        z_exp = [math.exp(u_i) for u_i in u]
        sum_z_exp = sum(z_exp)
        return [i / sum_z_exp for i in z_exp]

    else:
        raise TypeError("Expected a list or a list of lists")
