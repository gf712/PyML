import unittest
from pyml.linear_models import LinearRegression, LogisticRegression
from pyml.datasets import regression, gaussian
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

    def test_mae(self):
        self.assertAlmostEqual(self.regressor.score(self.X_test, self.y_test, scorer='mae'),
                               0.9126392424298799, delta=0.001)

    def test_seed(self):
        self.assertEqual(self.regressor.seed, 1970)

    def test_solver_error(self):
        self.assertRaises(ValueError, LinearRegression, 1970, True, 'adadelta')


class LinearRegressionOLSTest(unittest.TestCase):

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


class LogisticRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = X, y = gaussian(labels=2, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970)
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_log_iterations(self):
        self.assertEqual(self.classifier.iterations, 1623)

    def test_log_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0], -1.1576475345638408, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1], 0.1437129269620468, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2], 2.4464052394504856, delta=0.001)

    def test_log_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0], -106.11158912690777, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[-1], -61.16035391042087, delta=0.001)

    def test_log_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_log_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0], 0.807766417948826, delta=0.001)

    def test_log_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.975, delta=0.001)

    def test_log_seed(self):
        self.assertEqual(self.classifier.seed, 1970)


class MultiClassLogisticRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = X, y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970)
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_log_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 3829)
        self.assertEqual(self.classifier.iterations[1], 4778)
        self.assertEqual(self.classifier.iterations[2], 3400)

    def test_log_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -2.504659172303325, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 0.9999686753579901, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 0.5430990877594853, delta=0.001)

    def test_log_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -79.29189774967327, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -110.83233940996249, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -77.52658972786226, delta=0.001)

    def test_log_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_log_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.18176321188156466, delta=0.001)

    def test_log_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9666666666666667, delta=0.001)

    def test_log_seed(self):
        self.assertEqual(self.classifier.seed, 1970)


class MultiClassLogisticRegressionwithMomentumTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = X, y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970, alpha=0.9)
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_log_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 1671)
        self.assertEqual(self.classifier.iterations[1], 1691)
        self.assertEqual(self.classifier.iterations[2], 1546)

    def test_log_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -6.179813361986948, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 3.915365241814121, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 1.4391187417309603, delta=0.001)

    def test_log_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -38.81551959630421, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -71.12677749190144, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -39.45585327706164, delta=0.001)

    def test_log_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_log_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.04680865754859053, delta=0.001)

    def test_log_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)


class MultiClassLogisticRegressionMiniBatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls .X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970, alpha=0.9, batch_size=64)
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_log_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 1670)

    def test_log_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -6.166939294187147, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 3.9011202616018834, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 1.4360170566346704, delta=0.001)

    def test_log_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -38.882733287747826, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -71.22041849047173, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -39.508909436219405, delta=0.001)

    def test_log_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_log_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.047083994898380166, delta=0.001)

    def test_log_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)
