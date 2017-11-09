import unittest
from pyml.cluster import KMeans
from pyml.datasets import gaussian
from pyml.preprocessing import train_test_split


class MathsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datapoints, cls.labels = gaussian(n=100, d=2, labels=3, sigma=0.1, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.datapoints, cls.labels,
                                                                            train_split=0.95, seed=1970)
        cls.classifier = KMeans(k=3, seed=1970)
        cls.classifier.train(X=cls.X_train)

    def test_KMeansIterations(self):
        self.assertEqual(self.classifier.iterations, 9)

    def test_KMeansCentroids(self):
        self.assertAlmostEqual(self.classifier.centroids[1][1], 0.21926563270201577)

    def test_KMeansPredict(self):
        self.assertListEqual(self.classifier.predict(self.X_test), [0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 0, 1, 0, 0, 1])
