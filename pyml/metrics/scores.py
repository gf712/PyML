from pyml.maths.math_utils import mean


def mean_squared_error(y, y_true):
    """
    Mean squared error.

    Args:
        y (list): vector with predicted values.
        y_true (list): vector with true values.

    Returns:
        float: mean squared error.

    """
    return mean([(y_i - y_true_i) ** 2 for y_i, y_true_i in zip(y, y_true)])


def mean_absolute_error(y, y_true):
    """
    Mean absolute error.

    Args:
        y (list): vector with predicted values.
        y_true (list): vector with true values.

    Returns:
        float: mean absolute error.

    """
    return mean([abs(y_i - y_true_i) for y_i, y_true_i in zip(y, y_true)])


def accuracy(y, y_true):
    """
    Accuracy of class prediction.
    Args:
        y (list): vector with predicted labels.
        y_true (list): vector with true labels.

    Returns:
        float: accuracy.
    """
    return sum([1 for y_i, y_true_i in zip(y, y_true) if y_i == y_true_i]) / len(y)
