import unittest
from pyml.metrics.distances import euclidean_distance, manhattan_distance, calculate_distance
from pyml.metrics.scores import mean_absolute_error, mean_squared_error
from pyml.preprocessing import train_test_split
from pyml.utils import set_seed
import random
from pyml.datasets import regression
from pyml.nearest_neighbours import KNNRegressor


class DistancesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        set_seed(2017)

        cls.A = [[random.random() for e in range(3)] for x in range(3)]
        cls.B = [[random.random() for e in range(3)] for x in range(3)]

        cls.X, cls.y = regression(100, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.regressor = KNNRegressor(n=5)
        cls.regressor.train(X=cls.X_train, y=cls.y_train)

    def test_euclidean(self):
        self.assertListEqual(euclidean_distance(self.A, self.B), [0.9506105259861932,
                                                                  1.068591055912026,
                                                                  0.19746815659817757])

    def test_manhattan(self):
        self.assertListEqual(manhattan_distance(self.A, self.B), [1.390086685264639,
                                                                  1.6562536208811662,
                                                                  0.26064691033112874])

    def test_norm3(self):
        self.assertListEqual(calculate_distance(self.A, self.B, 3), [0.8460414137912776,
                                                                     0.9428389207901624,
                                                                     0.19178702833515607])

    def test_mse(self):
        self.assertAlmostEqual(mean_squared_error(self.regressor.predict(self.X_test), self.y_test),
                               1.5470835956432736)

    def test_mae(self):
        self.assertAlmostEqual(mean_absolute_error(self.regressor.predict(self.X_test), self.y_test),
                               1.024567537840727)

    def test_distance_error(self):
        self.assertRaises(ValueError, calculate_distance, self.regressor.predict(self.X_test), self.y_test,
                          "the_ultimate_norm")
