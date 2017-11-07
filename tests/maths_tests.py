import unittest
from pyml.maths.math_utils import *
from pyml.maths.linear_algebra import *
from pyml.utils import set_seed
import random


class MathsTest(unittest.TestCase):

    def test_sort(self):
        array = [-5, 3, 10, 2, 1, -1]
        sorted_array = sort(array)
        self.assertEqual(sorted_array, [-5, -1, 1, 2, 3, 10])

    def test_argsort(self):
        array = [-5, 3, 10, 2, 1, -1]
        argsorted_array = argsort(array)
        self.assertEqual(argsorted_array, [0, 4, 5, 3, 2, 1])


class LinearAlgebraTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        set_seed(1970)

        cls.A = [[random.random() for e in range(10)] for x in range(10)]

    def test_transpose(self):
        self.assertAlmostEqual(transpose(self.A)[5][8], 0.7441599310937185)
