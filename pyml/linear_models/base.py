from pyml.base import BaseLearner
from pyml.base import Predictor
import random


class LinearBase(BaseLearner, Predictor):

    """
    Base class for linear models
    """

    def __init__(self):
        """
        Inherits methods from BaseLearner
        """

        pass

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
