from .math_utils import mean


def dot_product(u, v):
    return sum([u_i * v_i for u_i, v_i in zip(u, v)])


def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def mean_squared_error(y, y_true):
    return mean([(y_i - y_true_i) ** 2 for y_i, y_true_i in zip(y, y_true)])
