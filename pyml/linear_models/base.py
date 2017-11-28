from pyml.base import BaseLearner
from pyml.base import Predictor
import random
from pyml.maths.optimisers import gradient_descent
from pyml.utils import set_seed


class LinearBase(BaseLearner, Predictor):

    """
    Base class for linear models
    """

    def __init__(self, learning_rate, epsilon, max_iterations, alpha, batch_size, method, seed, _type):
        """
        Inherits methods from BaseLearner
        """
        BaseLearner.__init__(self)
        Predictor.__init__(self)

        self._epsilon = epsilon
        self._max_iterations = max_iterations
        self._learning_rate = learning_rate

        self._alpha = alpha
        self._batch_size = batch_size

        if method in ['normal', 'nesterov']:
            self._method = method
        else:
            raise ValueError("Unknown GD method")

        self._seed = set_seed(seed)
        self._type = _type

    def _initiate_weights(self, bias):
        """
        initialisation of weights

        :type bias: bool
        :param bias: whether or not to include bias, and if so add a column of 1's
        :rtype: None
        :return: returns init coefficients
        """
        if bias:
            coefficients = [random.gauss(0, 1) for x in range(self._n_features + 1)]
            self.X = [[1] + row for row in self.X]
            return coefficients
        else:
            coefficients = [random.gauss(0, 1) for x in range(self._n_features)]
            return coefficients

    def _gradient_descent(self, X, y, theta):

        return gradient_descent(X, theta, y, self._batch_size, self._max_iterations, self._epsilon, self._learning_rate,
                                self._alpha, self._type, self._method, self._seed)
