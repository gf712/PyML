from pyml.maths.math_utils import argsort
from pyml.metrics.distances import calculate_distance
from pyml.base import BaseLearner


class KNNBase(BaseLearner):
    def __init__(self):
        """

        :param neighbours:
        :param norm:
        """
        BaseLearner.__init__(self)

    def _train(self, X, y):
        self.X = X
        self.y = y

    def _find_neighbours(self, X):
        """

        :param X:
        :param y:
        :return:
        """
        self._neighbours = list()

        # for each data point find n closest points in training set
        for x_i in X:
            distances_i = self._distance(self.X, x_i, self.norm)

            # get order of distances
            sorted_distances = argsort(distances_i)

            # get k points
            k_nearest_neighbours = [self.y[sorted_distances[0]] for i in range(self.n)]

            # majority vote
            self._neighbours.append(k_nearest_neighbours)

    @staticmethod
    def _distance(u, v, norm):
        """

        :param u:
        :param v:
        :param norm:
        :return:
        """
        if norm == 'l1':
            p = 1
        elif norm == 'l2':
            p = 2
        else:
            p = norm

        return calculate_distance(u, v, p)
