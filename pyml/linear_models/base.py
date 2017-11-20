from ..base import BaseLearner
from pyml.base import Predictor
import random


class LinearBase(BaseLearner, Predictor):

    def __init__(self):

        pass

    def _initiate_weights(self, bias):
        if bias:
            self._coefficients = [random.gauss(0, 1) for x in range(self._n_features + 1)]
            self.X = [[1] + row for row in self.X]
        else:
            self._coefficients = [random.gauss(0, 1) for x in range(self._n_features)]
