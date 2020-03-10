#
# Tests for the Variable class
#
import pybamm
import numpy as np

import unittest


class TestVariable(unittest.TestCase):
    def test_variable_init(self):
        a = pybamm.Variable("a")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.domain, [])
        a = pybamm.Variable("a", domain=["test"])
        self.assertEqual(a.domain[0], "test")
        self.assertRaises(TypeError, pybamm.Variable("a", domain="test"))

    def test_variable_id(self):
        a1 = pybamm.Variable("a", domain=["negative electrode"])
        a2 = pybamm.Variable("a", domain=["negative electrode"])
        self.assertEqual(a1.id, a2.id)
        a3 = pybamm.Variable("b", domain=["negative electrode"])
        a4 = pybamm.Variable("a", domain=["positive electrode"])
        self.assertNotEqual(a1.id, a3.id)
        self.assertNotEqual(a1.id, a4.id)


class TestExternalVariable(unittest.TestCase):
    def test_external_variable_scalar(self):
        a = pybamm.ExternalVariable("a", 1)
        self.assertEqual(a.size, 1)

        self.assertEqual(a.evaluate(u={"a": 3}), 3)

        with self.assertRaisesRegex(KeyError, "External variable"):
            a.evaluate()
        with self.assertRaisesRegex(TypeError, "inputs u"):
            a.evaluate(u="not a dictionary")

    def test_external_variable_vector(self):
        a = pybamm.ExternalVariable("a", 10)
        self.assertEqual(a.size, 10)

        a_test = 2 * np.ones((10, 1))
        np.testing.assert_array_equal(a.evaluate(u={"a": a_test}), a_test)

        np.testing.assert_array_equal(a.evaluate(u={"a": 2}), a_test)

        with self.assertRaisesRegex(ValueError, "External variable"):
            a.evaluate(u={"a": np.ones((5, 1))})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
