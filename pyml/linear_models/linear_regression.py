from .base import LinearBase
from ..maths import dot_product, least_squares, gradient_descent
from pyml.metrics.scores import mean_squared_error, mean_absolute_error
from ..utils import set_seed


class LinearRegression(LinearBase):
    def __init__(self, seed=None, bias=True, solver='OLS', learning_rate=0.01,
                 epsilon=0.01, max_iterations=10000):
        """

        :param seed:
        :param bias:
        :param solver:
        :param learning_rate:
        :param epsilon:
        :param max_iterations:
        """

        LinearBase.__init__(self)

        self._seed = set_seed(seed)
        self.bias = bias
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self._learning_rate = learning_rate
        if solver in ['OLS', 'gradient_descent']:
            self._solver = solver

    def _train(self, X, y=None):
        self.X = X
        self.y = y

        self._n_features = len(X[0])

        if self._solver == 'gradient_descent':
            self._initiate_weights(bias=self.bias)
            self._coefficients, self._cost, self._iterations = gradient_descent.gradient_descent(self.X,
                                                                                                 self.coefficients,
                                                                                                 self.y,
                                                                                                 self.max_iterations,
                                                                                                 self.epsilon,
                                                                                                 self._learning_rate,
                                                                                                 'rgrs')
        else:
            if self.bias:
                self.X = [[1] + row for row in self.X]
            self._cost = 'NaN'
            self._iterations = 'NaN'
            self._coefficients = least_squares(self.X, self.y)

    def _predict(self, X):

        if self.bias and len(X[0]) == self._n_features + 1:
            return dot_product(X, self.coefficients)
        elif self.bias and len(X[0]) == self._n_features:
            return dot_product([[1] + row for row in X], self._coefficients)
        elif not self.bias:
            return dot_product(X, self.coefficients)
        else:
            raise ValueError("Something went wrong.")

    def _score(self, X, y_true, scorer='mean_squared_error'):
        if scorer == 'mean_squared_error':
            return mean_squared_error(self.predict(X), y_true)
        elif scorer == 'mean_absolute_error':
            return mean_absolute_error(self.predict(X), y_true)

    @property
    def seed(self):
        return self._seed

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def cost(self):
        return self._cost

    @property
    def iterations(self):
        return self._iterations
