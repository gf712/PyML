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
