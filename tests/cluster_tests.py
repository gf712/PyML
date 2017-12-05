import unittest
from pyml.cluster import KMeans
from pyml.datasets import gaussian
from pyml.preprocessing import train_test_split


class KMeansTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datapoints, cls.labels = gaussian(n=100, d=2, labels=3, sigma=0.1, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.datapoints, cls.labels,
                                                                            train_split=0.95, seed=1970)
        cls.classifier = KMeans(k=3, seed=1970, norm=1)
        cls.classifier.train(X=cls.X_train)

    def test_KMeansIterations(self):
        self.assertEqual(self.classifier.iterations, 7)

    def test_KMeansCentroids(self):
        self.assertAlmostEqual(self.classifier.centroids[1][1], 0.21926563270201577)

    def test_KMeansPredict(self):
        self.assertListEqual(self.classifier.predict(self.X_test), [0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 0, 1, 0, 0, 1])

    def test_KMeansSeed(self):
        self.assertEqual(self.classifier.seed, 1970)

    def test_KMeansMinChange(self):
        self.assertEqual(self.classifier.min_change, 1)

    def test_KMeans_Init_Error(self):
        self.assertRaises(ValueError, KMeans, 1, 'foo')

    def test_KMeans_Norm_Error(self):
        self.assertRaises(ValueError, KMeans, 1, 'Forgy', 100, 1, None, 'amazing norm')

    def test_KMeans_MoreKThanN_Error(self):
        datapoints, labels = gaussian(n=5, d=2, labels=2, sigma=0.1, seed=1970)
        classifier = KMeans(k=11)
        self.assertRaises(ValueError, classifier.train, datapoints, labels)

    def test_KMeans_l2(self):
        datapoints, labels = gaussian(n=100, d=2, labels=3, sigma=0.1, seed=1970)
        X_train, y_train, X_test, y_test = train_test_split(datapoints, labels,
                                                            train_split=0.95, seed=1970)
        classifier = KMeans(k=3, seed=1970, norm='l2')
        classifier.train(X=X_train)
        self.assertEqual(self.classifier.iterations, 7)


    def test_KMeans_random_init(self):
        datapoints, labels = gaussian(n=100, d=2, labels=3, sigma=0.1, seed=1970)
        X_train, y_train, X_test, y_test = train_test_split(datapoints, labels,
                                                            train_split=0.95, seed=1970)
        classifier = KMeans(k=3, seed=1970, initialisation='Random')
        classifier.train(X=X_train)
        self.assertEqual(self.classifier.iterations, 7)
