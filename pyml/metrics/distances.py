def calculate_distance(u, v, p):
    if p == 1:
        return manhattan_distance(u, v)
    elif p == 2:
        return euclidean_distance(u, v)
    else:
        norm(u, v, p)


def norm(u, v, p):
    return sum([abs(u_i - v_i) ** p for u_i, v_i in zip(u, v)]) ** (1 / p)


def euclidean_distance(u, v):
    return norm(u, v, p=2)


def manhattan_distance(u, v):
    return norm(u, v, p=1)
