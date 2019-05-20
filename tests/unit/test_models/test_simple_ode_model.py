#
# Tests for the simple ODE model
#
import pybamm
import tests

import unittest
import numpy as np


class TestSimpleODEModel(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.SimpleODEModel()

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.SimpleODEModel()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)

    def test_solution(self):
        model = pybamm.SimpleODEModel()
        modeltest = tests.StandardModelTest(model)
        t_eval = np.linspace(0, 1, 50)
        modeltest.test_all(t_eval=t_eval)
        t, y = modeltest.solver.t, modeltest.solver.y
        mesh = modeltest.disc.mesh

        # check output
        processed_variables = pybamm.post_process_variables(model.variables, t, y, mesh)
        np.testing.assert_array_almost_equal(processed_variables["a"](t), 2 * t)
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = mesh.combine_submeshes(*whole_cell)[0].nodes
        np.testing.assert_array_almost_equal(
            processed_variables["b broadcasted"](t, x), np.ones((len(x), len(t)))
        )
        x_n_s = mesh.combine_submeshes("negative electrode", "separator")[0].nodes
        np.testing.assert_array_almost_equal(
            processed_variables["c broadcasted"](t, x_n_s),
            np.ones_like(x_n_s)[:, np.newaxis] * np.exp(-t),
        )

    def test_manufactured_solution(self):
        model = pybamm.SimpleODEModel()
        approx_exact_ns_dict = tests.get_manufactured_solution_errors(model, [1])
        # ODE model: the error should be almost zero for each var (no convergence test)
        for approx_exact in approx_exact_ns_dict.values():
            approx, exact = approx_exact[1]
            t = np.linspace(0, 1, 100)
            x = approx.x_sol
            errors = approx(t, x) - exact(t, x)
            np.testing.assert_almost_equal(errors, 0, decimal=5)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
