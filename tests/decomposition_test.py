import unittest
from pyml.decomposition import PCA
from pyml.datasets import load_iris


class PCATest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_iris()
        cls.decomposer = PCA()
        cls.decomposer.train(X=cls.X)

    def test_PCA_n_components(self):
        self.assertEqual(self.decomposer.n_components, 4)

    def test_PCA_eigenvalues(self):
        self.assertCountEqual(self.decomposer.eigenvalues, [4.1966751631979795, 0.24062861448333198,
                                                            0.07800041537352681, 0.023525140278494793])

    def test_PCA_eigenvectors(self):
        self.assertAlmostEqual(self.decomposer.eigenvectors[0][2], 0.5809972798275975)

    def test_PCA_transform(self):
        self.assertAlmostEqual(self.decomposer.transform(self.X)[1][2], 0.203521425006)

    def test_PCA_train_transform(self):
        self.assertAlmostEqual(self.decomposer.train_transform(self.X)[1][2], 0.203521425006)

    def test_PCA_inverse(self):
        R = self.decomposer.train_transform(self.X)
        self.assertAlmostEqual(self.decomposer.inverse(R)[1][2], self.X[1][2])
