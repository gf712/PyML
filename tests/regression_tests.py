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
        self.assertAlmostEqual(self.regressor.cost[-1], 0.4868770157376261, delta=0.001)

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
        self.assertAlmostEqual(self.classifier.cost[-1], -61.15035744417768, delta=0.001)

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
        self.assertAlmostEqual(self.classifier.cost[0][-1], -79.28190020206335, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -110.82234100438215, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -77.51659078552537, delta=0.001)

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
        self.assertAlmostEqual(self.classifier.cost[0][-1], -38.80552082812185, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -71.11678230563942, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -39.44585417456268, delta=0.001)

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
        self.assertEqual(self.classifier.iterations[0], 905)
        self.assertEqual(self.classifier.iterations[1], 789)
        self.assertEqual(self.classifier.iterations[2], 841)

    def test_MLogRMin_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -8.55590366737834, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 5.300981091494979, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 1.9547910473910273, delta=0.001)

    def test_MLogRMin_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -28.947086360092676, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -63.442959967464574, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -30.33741258448764, delta=0.001)

    def test_MLogRMin_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRMin_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.017049079187284634, delta=0.001)

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
        self.assertEqual(self.classifier.iterations[0], 1673)
        self.assertEqual(self.classifier.iterations[1], 1692)
        self.assertEqual(self.classifier.iterations[2], 1548)

    def test_MLogRNesOpt_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -6.180431021808369, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 3.914247513063426, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 1.4391579779616654, delta=0.001)

    def test_MLogRNesOpt_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -38.80233474838201, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -71.1246527942097, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -39.44375056521835, delta=0.001)

    def test_MLogRNesOpt_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRNesOpt_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.046812477405301665, delta=0.001)

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
        self.assertAlmostEqual(self.classifier.cost[0][-1], -33.083261965268775, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -60.5014103879351, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -26.87955154897393, delta=0.001)

    def test_MLogRAdagradOpt_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRAdagradOpt_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.01683721954111243, delta=0.001)

    def test_MLogRAdagradOpt_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)


class MultiClassLogisticRegressionAdadeltaOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls .X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970, learning_rate=1, alpha=0.93, method='adadelta')
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_MLogRAdagradOpt_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 50)
        self.assertEqual(self.classifier.iterations[1], 31)
        self.assertEqual(self.classifier.iterations[2], 59)

    def test_MLogRAdagradOpt_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -17.69287148773692, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 8.833657315503745, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 3.612736014955706, delta=0.001)

    def test_MLogRAdagradOpt_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -19.461391357431822, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -58.89195462566389, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -22.090057493027633, delta=0.001)

    def test_MLogRAdagradOpt_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRAdagradOpt_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.0008593472329257797, delta=0.001)

    def test_MLogRAdagradOpt_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)


class MultiClassLogisticRegressionRMSpropOpt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = gaussian(labels=3, sigma=0.2, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls .X, cls.y,
                                                                            train_split=0.8, seed=1970)
        cls.classifier = LogisticRegression(seed=1970, learning_rate=1, alpha=0.99, method='rmsprop')
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_MLogRAdagradOpt_iterations(self):
        self.assertEqual(self.classifier.iterations[0], 212)
        self.assertEqual(self.classifier.iterations[1], 32)
        self.assertEqual(self.classifier.iterations[2], 71)

    def test_MLogRAdagradOpt_coefficients(self):
        self.assertAlmostEqual(self.classifier.coefficients[0][-1], -15.450318804509942, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[1][-1], 9.072690798494117, delta=0.001)
        self.assertAlmostEqual(self.classifier.coefficients[2][-1], 4.915283731375461, delta=0.001)

    def test_MLogRAdagradOpt_cost(self):
        self.assertAlmostEqual(self.classifier.cost[0][-1], -19.841625448780448, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[1][-1], -58.50817514694194, delta=0.001)
        self.assertAlmostEqual(self.classifier.cost[2][-1], -21.77574545155869, delta=0.001)

    def test_MLogRAdagradOpt_predict(self):
        self.assertEqual(self.classifier.predict(self.X_test)[0], 1)

    def test_MLogRAdagradOpt_predict_proba(self):
        self.assertAlmostEqual(self.classifier.predict_proba(self.X_test)[0][0], 0.0009482861157384186, delta=0.001)

    def test_MLogRAdagradOpt_accuracy(self):
        self.assertAlmostEqual(self.classifier.score(self.X_test, self.y_test), 0.9833333333333333, delta=0.001)
