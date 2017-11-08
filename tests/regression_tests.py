import unittest
from pyml.linear_models import LinearRegression
from pyml.datasets import regression
from pyml.preprocessing import train_test_split


class LinearRegressionGradientDescentTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = regression(100, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.regressor = LinearRegression(seed=1970, solver='gradient_descent')
        cls.regressor.train(X=cls.X_train, y=cls.y_train)

    def test_iterations(self):
        self.assertEqual(self.regressor.iterations, 7)

    def test_coefficients(self):
        self.assertAlmostEqual(self.regressor.coefficients[0], 0.4907136205265401, delta=0.001)
        self.assertAlmostEqual(self.regressor.coefficients[1], 0.9034467828351432, delta=0.001)

    def test_cost(self):
        self.assertAlmostEqual(self.regressor.cost[0], 3.5181936893597365, delta=0.001)
        self.assertAlmostEqual(self.regressor.cost[-1], 0.49247697691721576, delta=0.001)

    def test_predict(self):
        self.assertAlmostEqual(self.regressor.predict(self.X_test)[0], 3.8176098320897065, delta=0.001)

    def test_mse(self):
        self.assertAlmostEqual(self.regressor.score(self.X_test, self.y_test), 1.3280324597827904, delta=0.001)


class LinearRegressionOLS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = regression(100, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.regressor = LinearRegression(seed=1970, solver='OLS')
        cls.regressor.train(X=cls.X_train, y=cls.y_train)

    def test_OLS_coefficients(self):
        self.assertAlmostEqual(self.regressor.coefficients[0], 0.518888884839874, delta=0.001)
        self.assertAlmostEqual(self.regressor.coefficients[1], 0.9128356664164721, delta=0.001)

    def test_OLS_predict(self):
        self.assertAlmostEqual(self.regressor.predict(self.X_test)[0], 3.880359176261411, delta=0.001)

    def test_OLS_mse(self):
        self.assertAlmostEqual(self.regressor.score(self.X_test, self.y_test), 1.34151578011058, delta=0.001)
