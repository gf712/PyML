from pyml.maths.math_utils import argsort
from pyml.metrics.distances import calculate_distance
from pyml.base import BaseLearner, Predictor


class KNNBase(BaseLearner, Predictor):
    def __init__(self):
        """
        Base learner for KNN regressor and classifier.
        """

        pass

    def _train(self, X, y):
        """
        KNN training phase is just retaining the training set

        Args:
            X (list): list of lists with each row corresponding to a
            datapoint's features
            y (list): list of targets
        """

        self.X = X
        self.y = y

    def _find_neighbours(self, X):
        """
        Finds the nearest neighbours to each point of X

        Args:
            X: list of lists with each row corresponding to a datapoint's
            features
        """

        self._neighbours = list()

        # for each data point find n closest points in training set
        for x_i in X:
            distances_i = calculate_distance(self.X, x_i, self.norm)

            # get order of distances
            sorted_distances = argsort(distances_i)

            # get k points
            k_nearest_neighbours = [self.y[sorted_distances[i]] for i in
                                    range(self.n)]

            # majority vote
            self._neighbours.append(k_nearest_neighbours)
