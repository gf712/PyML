def order(func, ascending=True):

    def wrapper(*args, **kwargs):
        array = func(*args, **kwargs)
        if ascending:
            return array
        else:
            return array[::-1]

    return wrapper
