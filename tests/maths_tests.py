import unittest
from pyml.maths.math_utils import *


class MathsTest(unittest.TestCase):

    def test_sort(self):
        array = [-5, 3, 10, 2, 1, -1]
        sorted_array = sort(array)
        self.assertEqual(sorted_array, [-5, -1, 1, 2, 3, 10])

    def test_argsort(self):
        array = [-5, 3, 10, 2, 1, -1]
        argsorted_array = argsort(array)
        self.assertEqual(argsorted_array, [0, 4, 5, 3, 2, 1])
