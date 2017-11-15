from ..base import BaseLearner
from ..maths import mean, transpose, argsort
from ..metrics.distances import euclidean_distance
import random


class ClusterBase(BaseLearner):

    def __init__(self):
        BaseLearner.__init__(self)

    def _get_cluster(self, cluster_label=0):
        """
        method to retrieve the index of all data points in a cluster
        :return: list
        """
        return [i for i, label in enumerate(self._labels) if label == cluster_label]

    def _initialise_centroids(self):

        if self._initialisation == 'Forgy':
            # choose k random datapoints to be centroids
            indices = set()
            while len(indices) < self.k:
                indices.add(random.randint(0, self.n))
            self._centroids = [self._X[x] for x in indices]

        elif self._initialisation == 'Random':
            self._centroids = [[random.random() for x in range(len(self._X[0]))] for x in range(self.k)]

        else:
            raise ValueError("Unknown initialisation.")

    def _get_cluster_mean(self, cluster_label=0):
        """
        Calculate centroid location
        :param cluster_label:
        :return: list
        """

        datapoints_index = self._get_cluster(cluster_label=cluster_label)
        datapoints = [self._X[i] for i in datapoints_index]
        # return the average of each dimension
        return mean(datapoints, axis=0)

    def _update_centroids(self):
        self._centroids = [self._get_cluster_mean(x) for x in range(self.k)]

    def _assign_cluster(self, X):
        # calculate distance to each centroid
        distances = [euclidean_distance(X, self._centroids[i]) for i in range(self.k)]
        # distances_T = transpose(distances)
        # # return label of closest cluster to each data point
        # return [x.index(min(x)) for x in distances_T]
        return [x[0] for x in argsort(distances, axis=0)]

    def _changes(self):
        change_n = 0
        for cluster in range(self.k):
            indices_set = set(self._get_cluster(cluster_label=cluster))
            old_indices = set(self._old_indices[cluster])
            change_n += len(old_indices - indices_set)
        return change_n
