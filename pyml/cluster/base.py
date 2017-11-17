from ..base import BaseLearner
from ..maths import mean, argsort
from ..metrics.distances import calculate_distance
import random


class ClusterBase(BaseLearner):

    """
    Base class for clustering.
    """

    def __init__(self):
        BaseLearner.__init__(self)

    def _get_cluster(self, cluster_label=0):
        """
        Method to retrieve the index of all data points in a cluster.
        Returns a list containing indices of datapoints belonging to cluster

        :param cluster_label: cluster label
        :type cluster_label: int
        :rtype: list
        :return: indices of datapoints belonging to cluster
        """
        return [i for i, label in enumerate(self._labels) if label == cluster_label]

    def _initialise_centroids(self):
        """
        Method to initialise centroids.
        Currently supports Forgy and Random initialisation
        (see https://en.wikipedia.org/wiki/K-means_clustering#Initialization_methods)
        This method directly sets centroids with the self._centroids attribute of the child class.

        :rtype: None
        """
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
        Calculate centroid location (which is the mean in each dimension of all points in a cluster).
        Returns a list with M coordinates for the cluster_label centroid.

        :param cluster_label: cluster number
        :type cluster_label: int
        :rtype: list
        :return: mean value of each dimension of cluster_label
        """

        datapoints_index = self._get_cluster(cluster_label=cluster_label)
        datapoints = [self._X[i] for i in datapoints_index]
        # return the average of each dimension
        return mean(datapoints, axis=0)

    def _assign_cluster(self, X):
        """
        Cluster assignment given the norm in the child class.
        Returns a list with the cluster labels (of size len(X))

        :param X: datapoints to be classified
        :type X: list of lists
        :rtype: list
        :return: cluster labels
        """

        # calculate distance to each centroid
        distances = [calculate_distance(X, self._centroids[i], self.norm) for i in range(self.k)]
        # distances_T = transpose(distances)
        # # return label of closest cluster to each data point
        # return [x.index(min(x)) for x in distances_T]
        return [x[0] for x in argsort(distances, axis=0)]

    def _changes(self):
        """
        Keeps track of the number of changes per iteration.
        :return: number of changes per iteration
        :rtype: int
        """
        change_n = 0
        for cluster in range(self.k):
            indices_set = set(self._get_cluster(cluster_label=cluster))
            old_indices = set(self._old_indices[cluster])
            change_n += len(old_indices - indices_set)
        return change_n
