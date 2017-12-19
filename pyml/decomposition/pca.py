from pyml.base import BaseLearner, Transformer
from pyml.maths import covariance, eigen, mean, dot_product, transpose, \
    subtract, add


class PCA(BaseLearner, Transformer):

    def __init__(self, n_components=0.95, tolerance=1.0e-9,
                 max_iterations=1000):

        """
        Principal component analysis.

        Args:
            n_components (int or float): Number of components to keep.
                If it is less than 1 it will be interpreted as a fraction of
                components to keep.
                Otherwise n_components will be kept.
            tolerance (float): Tolerance of jacobi rotations.
            max_iterations (int): maximum number of rotations.
        """

        BaseLearner.__init__(self)
        Transformer.__init__(self)

        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._n_components = n_components

    def _train(self, X, y=None):
        """
        Calculates the feature vector of X with PCA algorithm.
        Args:
            X (list): data to perform PCA on.
            y (NoneType): PCA does not use labeled data.
        """

        self._X = X
        self._n = len(X)
        self._m = len(X[0])

        if self._n_components < 1:
            self._n_components = int(round(self._n_components * self._m))

        if self._m > 20:
            raise NotImplementedError("Jacobi decomposition can be unstable "
                                      "with N>20 symmetric matrices!")

        # get the mean of each column
        self._X_means = mean(self._X, axis=0)

        # subtract each column by its mean
        X_whitened = subtract(self._X, self._X_means)

        # covariance matrix
        cov = covariance(X_whitened)

        # eigen decomposition of covariance matrix
        self._v, self._w = eigen(cov, self.tolerance, self.max_iterations,
                                 normalise=False, sort=True)

        # create feature vector
        self._feat_vect = [[self._w[row][column] for column in
                            range(self.n_components)] for row in
                           range(self._m)]

        return self

    def _transform(self, X):
        """
        Transform input matrix with feature vector and get PCA projections.
        Args:
            X (list): Input matrix.

        Returns:
            list: PCA projections.

        """
        # subtract X by mean
        X_whitened = subtract(X, self._X_means)

        return transpose(dot_product(transpose(self._feat_vect),
                                     transpose(X_whitened)))

    def _inverse(self, X):
        """
        Reverse transformation to original input matrix.

        Args:
            X (list): PCA projections.

        Returns:
            list: Input matrix.
        """

        return add(dot_product(X, transpose(self.eigenvectors)), self._X_means)

    @property
    def tolerance(self):
        """
        float: Returns the tolerance value for Jacobi rotations.
        """
        return self._tolerance

    @property
    def max_iterations(self):
        """
        int: Returns the maximum number of rotations of Jacobi rotations.
        """
        return self._max_iterations

    @property
    def eigenvalues(self):
        """
        list: Returns the eigenvalues in descending order.
        """
        return self._v

    @property
    def eigenvectors(self):
        """
        list: Returns the eigenvectors/ feature vectors.
        """
        return self._feat_vect

    @property
    def explained_variance(self):
        """
        list: Returns the explained variance of each component
        (it's the eigenvalue).
        """
        return self._v

    @property
    def explained_variance_ratio(self):
        """
        list: Returns the relative importance of each eigenvalue
        (sums up to 1).
        """
        return [eig / sum(self.eigenvalues) for eig in self.eigenvalues]

    @property
    def n_components(self):
        """
        int: Returns the number of components kept.
        """
        return self._n_components
