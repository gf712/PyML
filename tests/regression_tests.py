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
        self.assertEqual(self.regressor.iterations, 9)

    def test_coefficients(self):
        self.assertAlmostEqual(self.regressor.coefficients[0], 0.49321788756070506, delta=0.001)
        self.assertAlmostEqual(self.regressor.coefficients[1], 1.0352620298061765, delta=0.001)

    def test_cost(self):
        self.assertAlmostEqual(self.regressor.cost[0], 6.058113023701938, delta=0.001)
        self.assertAlmostEqual(self.regressor.cost[-1], 0.5841074648208295, delta=0.001)

    def test_predict(self):
        self.assertAlmostEqual(self.regressor.predict(self.X_test)[0], 3.0, delta=0.001)

    def test_mse(self):
        self.assertAlmostEqual(self.regressor.score(self.X_test, self.y_test), 1.4465282568357114, delta=0.001)
