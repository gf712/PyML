from .base import LinearBase
from ..maths import mean, dot_product, mean_squared_error, mean_absolute_error, subtract, power, transpose, divide
from ..utils import set_seed


class LinearRegression(LinearBase):

    def __init__(self, seed=None, bias=True, error_function='least_squares', learning_rate=0.01,
                 epsilon=0.01, max_iterations=10000):

        LinearBase.__init__(self)

        self._seed = set_seed(seed)
        self.bias = bias
        self._error_function = error_function
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self._learning_rate = learning_rate

    def _train(self, X, y=None):
        self.X = X
        self.y = y

        self._m = len(X)
        self._n_features = len(X[0])
        self._initiate_weights(bias=self.bias)
        self._cost = list()
        self._errors = list()

        e = 1000
        self._iteration = 0
        h = self._predict(self.X)
        loss = subtract(h, y)
        J_new = sum(power(loss, 2)) / (2 * self._m)
        X_transpose = transpose(self.X)

        # gradient descent
        while abs(e) >= self.epsilon and self._iteration < self.max_iterations:

            J_old = J_new
            self._cost.append(J_old)

            # calculate gradient for each feature
            gradients = divide(dot_product(X_transpose, loss), self._m)

            # update coefficients
            self._coefficients = [coefficient - self._learning_rate * gradient
                                  for coefficient, gradient in zip(self._coefficients, gradients)]

            h = self._predict(self.X)
            loss = subtract(h, y)
            J_new = sum(power(loss, 2)) / (2 * self._m)
            e = float(J_old - J_new)

            self._iteration += 1

    def _predict(self, X):
        # return [dot_product([1] + row, self._coefficients) if self.bias
        #         else dot_product(row, self._coefficients) for row in X]
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
        return self._iteration
