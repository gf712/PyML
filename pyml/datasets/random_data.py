import random
from ..utils import set_seed


def gaussian(n=100, d=2, labels=3, sigma=1, seed=None, shuffle=True):

    set_seed(seed)

    means = [[random.random() for dim in range(d)] for label in range(labels)]

    datapoints = list()
    data_labels = list()

    for label, mean in zip(range(labels), means):
        datapoints += [[random.gauss(mu=mean_i, sigma=sigma) for mean_i in mean] for x in range(n)]
        labels_i = [label] * n
        data_labels += labels_i

    if shuffle:
        data = list(zip(datapoints, data_labels))
        random.shuffle(data)
        datapoints, data_labels = zip(*data)

    return datapoints, data_labels


def regression(n=100, noise='gaussian', mu=0, sigma=1, x_min=[0], x_max=[10], intercept=0, gradient=[1], seed=None):

    set_seed(seed=seed)

    X = []

    for x_min_i, x_max_i in zip(x_min, x_max):
        X.append([x / x_max_i + random.random() for x in range(x_min_i, n)])

    y = list()
    for i in range(n):
        y_i = sum([x[i] * gradient_i for x, gradient_i in zip(X, gradient)])

        if noise == 'gaussian':
            e = random.gauss(mu=mu, sigma=sigma)

        y.append(y_i + e)

    X = [[x[i] for x in X] for i in range(n)]

    return X, y
