import random
from pyml.utils import set_seed
from pyml.preprocessing import shuffle_data


def gaussian(n=100, d=2, labels=3, sigma=1, seed=None, shuffle=True):

    """
    Builds a dummy dataset using a gaussian distribution.

    Args:
        n (int): number of datapoints per label
        d (int): number of dimensions
        labels (int): number of labels
        sigma (float): mean of distribution
        seed (int or NoneType): random seed
        shuffle (bool): whether to shuffle data

    Returns:
        tuple: (list with datapoints, labels of datapoints)
    """

    seed = set_seed(seed)

    means = [[random.random() for dim in range(d)] for label in range(labels)]

    datapoints = list()
    data_labels = list()

    for label, mean in zip(range(labels), means):
        datapoints += [[random.gauss(mu=mean_i, sigma=sigma) for mean_i
                        in mean] for x in range(n)]
        labels_i = [label] * n
        data_labels += labels_i

    if shuffle:
        datapoints, data_labels = shuffle_data(datapoints, data_labels)

    return datapoints, data_labels


def regression(n=100, noise='gaussian', mu=0, sigma=1, x_min=[0],
               x_max=[10], gradient=[1], seed=None):

    """
    Create a dummy dataset to perform regression. EXPERIMENTAL (can create a
    singular matrix)
    Args:
        n (int): number of datapoints
        noise (str): type of noise to add to the data.
            Currently only gaussian is supported.
        mu (float): mean of the noise.
        sigma (float): standard deviation of the noise.
        x_min (list): minimum value of each feature.
        x_max (list): maxium value of each feature.
        gradient (list): gradient of each feature.
        seed (int or NoneType): set random seed.

    Returns:
        tuple: (feature values, target values).

    """

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
