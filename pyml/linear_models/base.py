from pyml.base import BaseLearner
from pyml.base import Predictor
import random
from pyml.maths.optimisers import gradient_descent
from pyml.utils import set_seed
import warnings


class LinearBase(BaseLearner, Predictor):

    """
    """

    def __init__(self, learning_rate, epsilon, max_iterations, alpha,
                 fudge_factor, batch_size, method, seed, _type):
        """
        Base class for linear models.
        Inherits methods from BaseLearner and Predictor.

        Args:
            learning_rate (float): learning rate of gradient descent.
            epsilon (float): early stopping criterium for gradient descent.
                If the difference in loss of two consecutive iterations is
                less than delta the algorithm stops.
            max_iterations (int): maximum number of gradients descent
                iterations.
            alpha (float): momentum of gradient descent
            fudge_factor (float): fudge factor to prevent zero divisions.
            batch_size (int): batch size to perform batch gradients descent.
                Set to one to perform stochastic gradient descent
            method (str): gradient descent method.
                - 'normal'
                - 'nesterov'
                - 'adagrad'
                - 'adadelta'
                - 'rmsprop'
            seed (int or NoneType): set random seed
            _type (str): tells backend to perform classification or regression
                - 'regression'
                - 'classification'
        """

        BaseLearner.__init__(self)
        Predictor.__init__(self)

        self._epsilon = epsilon
        self._max_iterations = max_iterations
        self._learning_rate = learning_rate

        self._alpha = alpha
        self._batch_size = batch_size

        if method in ['normal', 'nesterov', 'adagrad', 'adadelta', 'rmsprop']:
            self._method = method

        else:
            raise ValueError("Unknown GD method")

        self._seed = set_seed(seed)
        self._type = _type

        if self._method == 'adagrad' or self._method == 'adadelta' or \
                self._method == 'rmsprop':
            if fudge_factor == 0:
                warnings.warn("Fudge factor for {} optimisation is 0, "
                              "it will be set to 10e-8 for your own "
                              "safety".format(self._method))
                fudge_factor = 10e-8

        if self._method == 'adadelta' and self._learning_rate != 1:
            warnings.warn("Adadelta does not use a learning rate, "
                          "setting this value to 1!")
            self._learning_rate = 1

        self._fudge_factor = fudge_factor

    def _initiate_weights(self, bias):

        """
        Weight initialisation method.

        Args:
            bias (bool): whether or not to include bias, and if so add a
                column of 1's.

        Returns (list): returns initialised coefficients.

        """

        if bias:
            coefficients = [random.gauss(0, 1) for x in
                            range(self._n_features + 1)]
            self.X = [[1] + row for row in self.X]
            return coefficients
        else:
            coefficients = [random.gauss(0, 1) for x in
                            range(self._n_features)]
            return coefficients

    def _gradient_descent(self, X, y, theta):

        return gradient_descent(X, theta, y, self._batch_size,
                                self._max_iterations, self._epsilon,
                                self._learning_rate, self._alpha, self._type,
                                self._method, self._seed, self._fudge_factor)
