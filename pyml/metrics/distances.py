from .CMetrics import norm


def calculate_distance(u, v, p):
    if p == 1:
        return manhattan_distance(u, v)
    elif p == 2:
        return euclidean_distance(u, v)
    else:
        norm(u, v, p)


def euclidean_distance(u, v):
    return norm(u, v, 2)


def manhattan_distance(u, v):
    return norm(u, v, 1)
