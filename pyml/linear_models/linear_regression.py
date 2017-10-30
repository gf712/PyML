from .base import LinearBase
from ..maths import mean, dot_product, mean_squared_error, mean_absolute_error
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
        self._cost = list()
        self._errors = list()

    def _train(self, X, y=None):
        self.X = X
        self.y = y

        self._n_features = len(X[0])
        self._initiate_weights(bias=self.bias)

        e = 1000
        self._iteration = 0
        J_new = mean_squared_error(self._predict(self.X), self.y)
        prediction = self._predict(X)

        # gradient descent
        while abs(e) >= self.epsilon and self._iteration < self.max_iterations:

            J_old = J_new
            self._cost.append(J_old)

            # calculate gradient for each feature
            gradients = list()
            for j in range(len(self._coefficients)):
                gradients.append(mean([(prediction_i - y_i) * x_i[j]
                                       for x_i, prediction_i, y_i in zip(self.X, prediction, self.y)]))

            # update coefficients
            self._coefficients = [coefficient - self._learning_rate * gradient
                                  for coefficient, gradient in zip(self._coefficients, gradients)]

            prediction = self._predict(X)
            J_new = mean_squared_error(prediction, self.y)
            e = float(J_old - J_new)

            self._iteration += 1

    def _predict(self, X):
        return [dot_product([1] + row, self._coefficients) if self.bias
                else dot_product(row, self._coefficients) for row in X]

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
