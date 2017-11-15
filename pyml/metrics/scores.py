from pyml.maths.math_utils import mean


def mean_squared_error(y, y_true):
    return mean([(y_i - y_true_i) ** 2 for y_i, y_true_i in zip(y, y_true)])


def mean_absolute_error(y, y_true):
    return mean([abs(y_i - y_true_i) for y_i, y_true_i in zip(y, y_true)])


def accuracy(y, y_true):
    return sum([1 for y_i, y_true_i in zip(y, y_true) if y_i == y_true_i]) / len(y)
