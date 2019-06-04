#
# Tests for the lead-acid LOQS model
#
import pybamm
import tests

import unittest
import numpy as np


class TestLeadAcidLOQS(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.lead_acid.LOQS()
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.lead_acid.LOQS()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        simp_and_python = optimtest.evaluate_model(simplify=True, to_python=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)
        np.testing.assert_array_almost_equal(original, simp_and_python)

    def test_charge(self):
        model = pybamm.lead_acid.LOQS()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current [A]": -1})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_zero_current(self):
        model = pybamm.lead_acid.LOQS()
        parameter_values = model.default_parameter_values
        parameter_values.update({"Typical current [A]": 0})
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

    def test_basic_processing_with_convection(self):
        model = pybamm.lead_acid.LOQS({"convection": True})
        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_manufactured_solution(self):
        model = pybamm.lead_acid.LOQS()
        exact, approx = tests.get_manufactured_solution_errors(model, False)
        # ODE model: the error should be almost zero for each var (no convergence test)
        for var in exact.keys():
            np.testing.assert_almost_equal(
                exact[var].entries - approx[var].entries, 0, decimal=6
            )
        # approx_exact_ns_dict = tests.get_manufactured_solution_errors(model, [1])
        # # ODE model: the error should be almost zero for each var (no convergence test)
        # for approx_exact in approx_exact_ns_dict.values():
        #     approx, exact = approx_exact[1]
        #     t = np.linspace(0, 1, 100)
        #     x = approx.x_sol
        #     errors = approx(t, x) - exact(t, x)
        #     np.testing.assert_almost_equal(errors, 0, decimal=5)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
