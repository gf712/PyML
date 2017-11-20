from pyml.base import BaseLearner
import random


class LinearBase(BaseLearner):

    """
    Base class for linear models
    """

    def __init__(self):
        """
        Inherits methods from BaseLearner
        """

        BaseLearner.__init__(self)

    def _initiate_weights(self, bias):
        """
        initialisation of weights

        :type bias: bool
        :param bias: whether or not to include bias, and if so add a column of 1's
        :rtype: None
        :return: directly interacts with child class
        """
        if bias:
            self._coefficients = [random.gauss(0, 1) for x in range(self._n_features + 1)]
            self.X = [[1] + row for row in self.X]
        else:
            self._coefficients = [random.gauss(0, 1) for x in range(self._n_features)]
