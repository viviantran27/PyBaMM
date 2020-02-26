#
# Tests for the Matrix class
#
import pybamm
import numpy as np

import unittest


class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.A = np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1]])
        self.x = np.array([1, 2, 3])
        self.mat = pybamm.Matrix(self.A)
        self.vect = pybamm.Vector(self.x)

    def test_array_wrapper(self):
        self.assertEqual(self.mat.ndim, 2)
        self.assertEqual(self.mat.shape, (3, 3))
        self.assertEqual(self.mat.size, 9)

    def test_matrix_evaluate(self):
        np.testing.assert_array_equal(self.mat.evaluate(), self.A)

    def test_matrix_operations(self):
        np.testing.assert_array_equal((self.mat + self.mat).evaluate(), 2 * self.A)
        np.testing.assert_array_equal(
            (self.mat - self.mat).evaluate().toarray(), 0 * self.A
        )
        np.testing.assert_array_equal(
            (self.mat @ self.vect).evaluate(), np.array([[5], [2], [3]])
        )


class TestArray(unittest.TestCase):
    def test_name(self):
        arr = pybamm.Array(np.array([1, 2, 3]))
        self.assertEqual(arr.name, "Array of shape (3, 1)")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
