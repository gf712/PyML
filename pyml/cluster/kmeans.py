from pyml.cluster.base import ClusterBase
from pyml.utils import set_seed
import warnings


class KMeans(ClusterBase):
    def __init__(self, k=3, initialisation='forgy', max_iterations=100, min_change=1,
                 seed=None, norm='l1'):
        """
        KMeans implementation

        Args:
            k (int): number of clusters.
            initialisation (str): indicates method to initialise clusters (currently forgy or random).
            max_iterations (int): maximum number of iterations.
            min_change (int): minimum assignment changes after each iteration required to continue algorithm.
            seed (int or NoneType): sets random seed.
            norm (str): norm to use in the calculation of distances between each point and all centroids,
                 e.g. 'l2' or 2 are equivalent to using the euclidean distance.

        Examples:
            >>> from pyml.cluster import KMeans
            >>> from pyml.datasets.random_data import gaussian
            >>> from pyml.preprocessing import train_test_split
            >>> datapoints, labels = gaussian(n=100, d=2, labels=3, sigma=0.1, seed=1970)
            >>> X_train, y_train, X_test, y_test = train_test_split(datapoints, labels, train_split=0.95, seed=1970)
            >>> kmeans = KMeans(k=3, max_iterations=1000, seed=1970)
            >>> _ = kmeans.train(X_train, y_train)
            >>> kmeans.iterations
            7
            >>> kmeans.centroids[1]
            [0.12801075816403754, 0.21926563270201577]
        """
        ClusterBase.__init__(self)

        if isinstance(norm, int) or norm in ['l1', 'l2']:
            self._norm = norm
        else:
            raise ValueError("Unknown norm.")

        self._seed = set_seed(seed)
        self._k = k
        self._max_iterations = max_iterations
        self._min_change = min_change

        if initialisation in ['forgy', 'random']:
            self._initialisation = initialisation
        else:
            raise ValueError("Unknown initialisation method.")

    def _train(self, X, y=None):
        """
        KMeans clustering to determine the position of centroids and cluster assignment given number of clusters (k).

        Args:
            X (list): list of size N of lists (all of size M) to perform KMeans on
            y (NoneType): KMeans does not use labels

        Notes:
            Algorithm:
                1. Initiate centroid coordinates
                2. Assign cluster labels to each point of X
                3. Update centroid coordinates of each cluster (average of each dimension of all point in a cluster)
                4. Repeat 2 and 3 until reaching one of the stopping criteria

        :type X: list
        :type y: None
        :param X: 
        :param y: None
        :rtype: object
        :return: self
        """
        self._X = X
        self._y = y
        self._n = len(X)

        if self.n < self.k:
            raise ValueError("Number of clusters should be lower than the number of data points, "
                             "instead got {} datapoints for {} clusters".format(self.n, self.k))

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
            warnings.warn("Failed to converge within {} iterations, consider increasing max_iterations".
                          format(self.max_iterations))

    def _predict(self, X):
        """
        Predict cluster assignment using centroids from training step

        :param X: list of size N of lists (all of size M) to perform prediction
        :rtype: list
        :return: list of label predictions
        """
        return self._assign_cluster(X)

    def _update_centroids(self):
        """
        Update rule of KMeans.
        Directly updates _centroids attribute
        :return: None
        """
        self._centroids = [self._get_cluster_mean(x) for x in range(self.k)]

    @property
    def k(self):
        """
        Number of clusters
        :getter: Returns the number of clusters k
        :type: int
        """
        return self._k

    @property
    def n(self):
        """
        Number of training examples
        :getter: Returns the number of training examples
        :type: int
        """
        return self._n

    @property
    def iterations(self):
        """
        Number of iterations of KMeans (if train method has been called)
        :getter: Returns the number of KMeans algorithm iterations
        :type: int
        """
        return self._iterations

    @property
    def max_iterations(self):
        """
        Maximum number of iterations to run KMeans for.
        :getter: Returns the maximum number of iterations
        :type: int
        """
        return self._max_iterations

    @property
    def min_change(self):
        """
        Minimum label changes per iteration.
        :getter: Returns the minimum number of changes per iteration
        :type: int
        """
        return self._min_change

    @property
    def seed(self):
        """
        Random seed.
        :getter: Returns the random seed number.
        :type: int
        """
        return self._seed

    @property
    def centroids(self):
        """
        List of centroid coordinates
        :getter: Returns a list of lists with centroid coordinates
        :type: list
        """
        return self._centroids

    @property
    def norm(self):
        """
        Norm for distance calculation
        :getter: Returns the norm used for distance calculations
        :type: int
        """
        return self._norm

