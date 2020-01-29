#
# Tests the utility functions.
#
import numpy as np
import os
import pybamm
import unittest


class TestUtil(unittest.TestCase):
    """
    Test the functionality in util.py
    """

    def test_load_function(self):
        # Test filename ends in '.py'
        with self.assertRaisesRegex(
            ValueError, "Expected filename.py, but got doesnotendindotpy"
        ):
            pybamm.load_function("doesnotendindotpy")

        # Test exception if absolute file not found
        with self.assertRaisesRegex(
            ValueError, "is an absolute path, but the file is not found"
        ):
            nonexistent_abs_file = os.path.join(os.getcwd(), "i_dont_exist.py")
            pybamm.load_function(nonexistent_abs_file)

        # Test exception if relative file not found
        with self.assertRaisesRegex(
            ValueError, "cannot be found in the PyBaMM directory"
        ):
            pybamm.load_function("i_dont_exist.py")

        # Test exception if relative file found more than once
        with self.assertRaisesRegex(
            ValueError, "found multiple times in the PyBaMM directory"
        ):
            pybamm.load_function("__init__.py")

        # Test exception if no matching function found in module
        with self.assertRaisesRegex(ValueError, "No function .+ found in module .+"):
            pybamm.load_function("process_symbol_bad_function.py")

        # Test function load with absolute path
        abs_test_path = os.path.join(
            os.getcwd(),
            "tests",
            "unit",
            "test_parameters",
            "data",
            "process_symbol_test_function.py",
        )
        self.assertTrue(os.path.isfile(abs_test_path))
        func = pybamm.load_function(abs_test_path)
        self.assertEqual(func(2), 246)

        # Test function load with relative path
        func = pybamm.load_function("process_symbol_test_function.py")
        self.assertEqual(func(3), 369)

    def test_rmse(self):
        self.assertEqual(pybamm.rmse(np.ones(5), np.zeros(5)), 1)
        self.assertEqual(pybamm.rmse(2 * np.ones(5), np.zeros(5)), 2)
        self.assertEqual(pybamm.rmse(2 * np.ones(5), np.ones(5)), 1)

        x = np.array([1, 2, 3, 4, 5])
        self.assertEqual(pybamm.rmse(x, x), 0)

        with self.assertRaisesRegex(ValueError, "same length"):
            pybamm.rmse(np.ones(5), np.zeros(3))

    def test_infinite_nested_dict(self):
        d = pybamm.get_infinite_nested_dict()
        d[1][2][3] = "x"
        self.assertEqual(d[1][2][3], "x")
        d[4][5] = "y"
        self.assertEqual(d[4][5], "y")

    def test_fuzzy_dict(self):
        d = pybamm.FuzzyDict({"test": 1, "test2": 2})
        self.assertEqual(d["test"], 1)
        with self.assertRaisesRegex(KeyError, "'test3' not found. Best matches are "):
            d["test3"]


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
