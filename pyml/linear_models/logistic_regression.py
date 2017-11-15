from .base import LinearBase
from ..base import Classifier
from ..maths import dot_product, gradient_descent, sigmoid
from pyml.metrics.scores import accuracy
from ..utils import set_seed


class LogisticRegression(LinearBase, Classifier):
    def __init__(self, seed=None, bias=True, learning_rate=0.01, epsilon=0.01, max_iterations=10000):
        """

        :param seed:
        :param bias:
        :param solver:
        :param learning_rate:
        :param epsilon:
        :param max_iterations:
        """

        LinearBase.__init__(self)
        Classifier.__init__(self)

        self._seed = set_seed(seed)
        self.bias = bias
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self._learning_rate = learning_rate

    def _train(self, X, y=None):
        self.X = X
        self.y = y

        self._n_features = len(X[0])

        self._initiate_weights(bias=self.bias)
        self._coefficients, self._cost, self._iterations = gradient_descent.gradient_descent(self.X,
                                                                                             self.coefficients,
                                                                                             self.y,
                                                                                             self.max_iterations,
                                                                                             self.epsilon,
                                                                                             self._learning_rate,
                                                                                             'logit')

    def _predict(self, X):
        if self.bias and len(X[0]) == self._n_features + 1:
            return [int(round(x)) for x in sigmoid(dot_product(X, self.coefficients))]
        elif self.bias and len(X[0]) == self._n_features:
            return [int(round(x)) for x in sigmoid(dot_product([[1] + row for row in X], self._coefficients))]
        elif not self.bias:
            return [int(round(x)) for x in sigmoid(dot_product(X, self.coefficients))]
        else:
            raise ValueError("Something went wrong.")

    def _predict_proba(self, X):
        if self.bias and len(X[0]) == self._n_features + 1:
            return sigmoid(dot_product(X, self.coefficients))
        elif self.bias and len(X[0]) == self._n_features:
            return sigmoid(dot_product([[1] + row for row in X], self._coefficients))
        elif not self.bias:
            return sigmoid(dot_product(X, self.coefficients))
        else:
            raise ValueError("Something went wrong.")

    def _score(self, X, y_true, scorer='accuracy'):
        if scorer == 'accuracy':
            return accuracy(self.predict(X), y_true)

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
