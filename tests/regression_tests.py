import unittest
from pyml.linear_models import LinearRegression, LogisticRegression
from pyml.linear_models.base import LinearBase
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

    def test_LinR_iterations(self):
        self.assertEqual(self.regressor.iterations, 7)

    def test_LinR_coefficients(self):
        self.assertAlmostEqual(self.regressor.coefficients[0], 0.4907136205265401, delta=0.001)
        self.assertAlmostEqual(self.regressor.coefficients[1], 0.9034467828351432, delta=0.001)

    def test_LinR_cost(self):
        self.assertAlmostEqual(self.regressor.cost[0], 3.5181936893597365, delta=0.001)
        self.assertAlmostEqual(self.regressor.cost[-1], 0.49247697691721576, delta=0.001)

    def test_LinR_predict(self):
        self.assertAlmostEqual(self.regressor.predict(self.X_test)[0], 3.8176098320897065, delta=0.001)

    def test_LinR_train_predict(self):
        self.assertAlmostEqual(self.regressor.train_predict(self.X_train, self.y_train)[0], 9.770956237446251,
                               delta=0.001)

    def test_LinR_mse(self):
        self.assertAlmostEqual(self.regressor.score(self.X_test, self.y_test), 1.3280324597827904, delta=0.001)

    def test_LinR_mae(self):
        self.assertAlmostEqual(self.regressor.score(self.X_test, self.y_test, scorer='mae'),
                               0.9126392424298799, delta=0.001)

    def test_LinR_seed(self):
        self.assertEqual(self.regressor.seed, 1970)

    def test_LinR_solver_error(self):
        self.assertRaises(ValueError, LinearRegression, 1970, True, 'unknown_solver')


class GradientDescentTest(unittest.TestCase):

    def test_GD_opt_InitError(self):
        self.assertRaises(ValueError, LinearBase, 0.01, 0.01, 10, 0.9, 64, 0.1, 'amazing_optimiser_algo', None,
                          'regressor')


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

    def test_LogR_iterations(self):
        self.assertEqual(self.classifier.iterations, 1623)

    def test_LogR_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0], -1.1576475345638408, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1], 0.1437129269620468, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2], 2.4464052394504856, delta=0.001)

    def test_LogR_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0], -106.11158912690777, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[-1], -61.16035391042087, delta=0.001)

    def test_LogR_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_LogR_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0], 0.807766417948826, delta=0.001)

    def test_LogR_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.975, delta=0.001)

    def test_LogR_seed(self):
        self.assertEqual(self.classifier.seed, 1970)


class MultiClassLogisticRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = X, y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970)
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_MLogR_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 3829)
        self.assertEqual(self.classifier.iterations[1], 4778)
        self.assertEqual(self.classifier.iterations[2], 3400)

    def test_MLogR_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -2.504659172303325, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 0.9999686753579901, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 0.5430990877594853, delta=0.001)

    def test_MLogR_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -79.29189774967327, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -110.83233940996249, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -77.52658972786226, delta=0.001)

    def test_MLogR_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogR_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.18176321188156466, delta=0.001)

    def test_MLogR_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9666666666666667, delta=0.001)

    def test_MLogR_seed(self):
        self.assertEqual(self.classifier.seed, 1970)


class MultiClassLogisticRegressionwithMomentumTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = X, y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970, alpha=0.9)
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_MLogRMom_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 1671)
        self.assertEqual(self.classifier.iterations[1], 1691)
        self.assertEqual(self.classifier.iterations[2], 1546)

    def test_MLogRMom_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -6.179813361986948, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 3.915365241814121, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 1.4391187417309603, delta=0.001)

    def test_MLogRMom_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -38.81551959630421, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -71.12677749190144, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -39.45585327706164, delta=0.001)

    def test_MLogRMom_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRMom_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.04680865754859053, delta=0.001)

    def test_MLogRMom_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)


class MultiClassLogisticRegressionMiniBatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls .X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970, alpha=0.9, batch_size=64)
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_MLogRMin_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 1670)

    def test_MLogRMin_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -6.166939294187147, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 3.9011202616018834, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 1.4360170566346704, delta=0.001)

    def test_MLogRMin_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -38.882733287747826, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -71.22041849047173, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -39.508909436219405, delta=0.001)

    def test_MLogRMin_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRMin_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.047083994898380166, delta=0.001)

    def test_MLogRMin_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)


class MultiClassLogisticRegressionNesterovOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls .X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970, alpha=0.9, method='nesterov')
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_MLogRNesOpt_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 1274)
        self.assertEqual(self.classifier.iterations[1], 1210)
        self.assertEqual(self.classifier.iterations[2], 1175)

    def test_MLogRNesOpt_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -7.298649513828892, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 4.610707704248638, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 1.690684230026362, delta=0.001)

    def test_MLogRNesOpt_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -33.32798686899964, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -66.63780970485656, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -34.41686105857577, delta=0.001)

    def test_MLogRNesOpt_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRNesOpt_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.02931216446502055, delta=0.001)

    def test_MLogRNesOpt_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)


class MultiClassLogisticRegressionAdagradOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls .X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970, alpha=0.98, method='adagrad')
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_MLogRAdagradOpt_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 1187)
        self.assertEqual(self.classifier.iterations[1], 317)
        self.assertEqual(self.classifier.iterations[2], 436)

    def test_MLogRAdagradOpt_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -7.3539416411807474, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 6.203333163198617, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 3.2839577904188673, delta=0.001)

    def test_MLogRAdagradOpt_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -33.093253127772456, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -60.51140413468329, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -26.88953190063318, delta=0.001)

    def test_MLogRAdagradOpt_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRAdagradOpt_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.01683721954111243, delta=0.001)

    def test_MLogRAdagradOpt_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)
