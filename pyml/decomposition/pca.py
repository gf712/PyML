from pyml.base import BaseLearner, Transformer
from pyml.maths import covariance, eigen, mean, dot_product, transpose, subtract, add


class PCA(BaseLearner, Transformer):
    """
    Principal component analysis

    """

    def __init__(self, n_components=0.95, tolerance=1.0e-9, max_iterations=1000):

        BaseLearner.__init__(self)
        Transformer.__init__(self)

        self._n_components = n_components
        self._tolerance = tolerance
        self._max_iterations = max_iterations

    def _train(self, X, y=None):
        """
        Calculates the feature vector of X with PCA algorithm

        """
        self._X = X
        self._n = len(X)
        self._m = len(X[0])

        if self._m > 20:
            raise NotImplementedError("Jacobi decomposition can be unstable with N>20 symmetric matrices!")

        if isinstance(self._n_components, float):
            self._n_components = int(round(self._n_components * self._m))

        # get the mean of each column
        self._X_means = mean(self._X, axis=0)

        # subtract each column by its mean
        X_whitened = subtract(self._X, self._X_means)

        # covariance matrix
        cov = covariance(X_whitened)

        # eigen decomposition of covariance matrix
        self._v, self._w = eigen(cov, self.tolerance, self.max_iterations, normalise=False, sort=True)

        # create feature vector
        self._feat_vect = [[self._w[row][column] for column in range(self.n_components)] for row in range(self._m)]

        return self

    def _transform(self, X):
        """

        :param X:
        :param y:
        :return:
        """
        # subtract X by mean
        X_whitened = subtract(X, self._X_means)

        return transpose(dot_product(transpose(self._feat_vect), transpose(X_whitened)))

    def _inverse(self, X):
        """

        :return:
        """

        return add(dot_product(X, transpose(self.eigenvectors)), self._X_means)

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def eigenvalues(self):
        return self._v

    @property
    def eigenvectors(self):
        return self._feat_vect

    @property
    def explained_variance(self):
        return self._v

    @property
    def explained_variance_ratio(self):
        return [eig / sum(self.eigenvalues) for eig in self.eigenvalues]

    @property
    def n_components(self):
        """
        :rtype: int
        :return:
        """
        return self._n_components
