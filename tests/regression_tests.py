import unittest
from pyml.linear_models import LinearRegression
from pyml.datasets import regression
from pyml.preprocessing import train_test_split


class LinearRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = regression(100, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.regressor = LinearRegression(seed=1970)
        cls.regressor.train(X=cls.X_train, y=cls.y_train)

    def test_iterations(self):
        self.assertEqual(self.regressor.iterations, 8)

    def test_coefficients(self):
        self.assertAlmostEqual(self.regressor.coefficients, [0.49150589812917345, 0.9085120784208242])

    def test_cost(self):
        self.assertAlmostEqual(self.regressor.cost[0], 28.153932606643423)
        self.assertAlmostEqual(self.regressor.cost[-1], 0.9737540314752522)

    def test_predict(self):
        self.assertAlmostEqual(self.regressor.predict(self.X_test)[0], 3.83705479960681)

    def test_mse(self):
        self.assertAlmostEqual(self.regressor.score(self.X_test, self.y_test), 1.3268518706533787)
