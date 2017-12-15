from pyml.utils import set_seed
import random


def train_test_split(X, y, train_split=0.3, shuffle=True, seed=None):

    """
    Train test split for model validation.

    Args:
        X (list): list of lists with features
        y (list): list of targets
        train_split (float): train split ratio
        shuffle (bool): whether or not to additionally shuffle the data
        seed (int or NoneType): random seed for shuffle

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    set_seed(seed)

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
