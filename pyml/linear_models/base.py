from ..base import BaseLearner
import random


class LinearBase(BaseLearner):

    def __init__(self):

        BaseLearner.__init__(self)

    def _initiate_weights(self, bias):
        if bias:
            self._coefficients = [random.gauss(0, 1) for x in range(self._n_features + 1)]
            self.X = [[1] + row for row in self.X]
        else:
            self._coefficients = [random.gauss(0, 1) for x in range(self._n_features)]