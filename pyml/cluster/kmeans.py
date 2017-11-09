from .base import ClusterBase
from ..utils import set_seed


class KMeans(ClusterBase):
    def __init__(self, k=3, initialisation='Forgy', max_iterations=100, min_change=1,
                 seed=None):

        """
        KMeans implementation

        :param k: str number of clusters
        :param initialisation: str method to initialise clusters (currently Forgy or Random)
        :param max_iterations: int maximum number of iterations
        :param min_change: int minimum assignment changes after each iteration required to continue algorithm
        """
        ClusterBase.__init__(self)

        self._seed = set_seed(seed)
        self._k = k
        self._max_iterations = max_iterations
        self._min_change = min_change

        if initialisation in ['Forgy', 'Random']:
            self._initialisation = initialisation
        else:
            raise ValueError("Unknown initialisation method.")

    def _train(self, X, y=None):
        self._X = X
        self._y = y
        self._n = len(X)
        self._initialise_centroids()
        self._dimensions = len(X[0])
        self._iterations = 0
        self._cluster_assignment = []
        change = self.n

        self._labels = self._assign_cluster(self._X)

        while self.iterations < self.max_iterations and self._min_change < change:
            self._old_indices = [self._get_cluster(i) for i in range(self.k)]
            self._update_centroids()
            self._labels = self._assign_cluster(self._X)
            self._iterations += 1
            change = self._changes()

        if change > self._min_change:
            print("Failed to converge within {} iterations, consider increasing max_iterations".format(
                self.max_iterations))

    def _predict(self, X):
        """
        Predict cluster assignment
        :param X:
        :return:
        """
        return self._assign_cluster(X)

    @property
    def k(self):
        return self._k

    @property
    def n(self):
        return self._n

    @property
    def iterations(self):
        return self._iterations

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def min_change(self):
        return self._min_change

    @property
    def seed(self):
        return self._seed

    @property
    def centroids(self):
        return self._centroids
