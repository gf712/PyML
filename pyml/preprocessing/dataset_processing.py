import random
import time


def train_test_split(X, y, train_split=0.3, shuffle=True, seed=None):

    if seed is None:
        seed = time.time()

    random.seed(seed)

    if shuffle:
        data = list(zip(X, y))
        random.shuffle(data)
        X, y = zip(*data)
        X = list(X)
        y = list(y)

    n_train = int(len(X) * train_split)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test
