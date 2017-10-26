import unittest
from pyml.nearest_neighbours import KNNClassifier
from pyml.datasets import gaussian
from pyml.preprocessing import train_test_split


class TestKNNClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.datapoints, cls.labels = gaussian(n=100, d=2, labels=3, sigma=0.1, seed=1970)
        cls.X_train, cls.y_train, cls.X_test, cls.y_test = train_test_split(cls.datapoints, cls.labels,
                                                                            train_split=0.95, seed=1970)
        cls.classifier = KNNClassifier(n=5)
        cls.classifier.train(X=cls.X_train, y=cls.y_train)

    def test_train(self):
        self.assertEqual(self.classifier.X, self.X_train)

    def test_predict(self):
        predictions = self.classifier.predict(X=self.X_test)
        self.assertEqual(predictions, [2, 2, 0, 0, 2, 0, 2, 2, 1, 1, 2, 0, 2, 2, 0])

    def test_score(self):
        accuracy = self.classifier.score(X=self.X_test, y_true=self.y_test)
        self.assertEqual(accuracy, 1.0)
