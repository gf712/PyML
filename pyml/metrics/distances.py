from .CMetrics import norm


def calculate_distance(u, v, p):

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
    return norm(u, v, 2)


def manhattan_distance(u, v):
    return norm(u, v, 1)
