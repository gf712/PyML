from .base import KNNBase
from pyml.base import Predictor, Classifier
from ..maths import max_occurence, mean
from pyml.metrics.scores import mean_squared_error, mean_absolute_error


class KNNClassifier(KNNBase, Classifier):
    def __init__(self, n=3, norm='l1'):
        KNNBase.__init__(self)
        self.n = n
        self.norm = norm

    def _predict(self, X):
        """
        Prediction using a majority classifier
        :return:
        """
        self._find_neighbours(X)
        return [max_occurence(row) for row in self._neighbours]

    def _score(self, X, y_true):
        """
        Calculate accuracy
        :param X:
        :param y:
        :return:
        """

        predictions = self.predict(X)

        return sum([1 if pred_i == y_i else 0 for pred_i, y_i in zip(predictions, y_true)]) / len(X)


class KNNRegressor(KNNBase):
    def __init__(self, n=3, norm='l1'):
        KNNBase.__init__(self)
        self.n = n
        self.norm = norm

    def _predict(self, X):
        """
        Prediction using the mean of neighbours
        :return:
        """
        self._find_neighbours(X)
        return [mean(row) for row in self._neighbours]

    def _score(self, X, y_true, scorer='mean_squared_error'):
        if scorer == 'mean_squared_error':
            return mean_squared_error(self.predict(X), y_true)
        elif scorer == 'mean_absolute_error':
            return mean_absolute_error(self.predict(X), y_true)
