#
# Tests for the Casadi Solver class
#
import casadi
import pybamm
import unittest
import numpy as np
from tests import get_mesh_for_testing
import warnings


class TestCasadiSolver(unittest.TestCase):
    def test_integrate(self):
        # Constant
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")

        y = casadi.SX.sym("y")
        constant_growth = casadi.SX(0.5)
        problem = {"x": y, "ode": constant_growth}

        y0 = np.array([0])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate_casadi(problem, y0, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(0.5 * solution.t, solution.y[0])

        # Exponential decay
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="cvodes")

        exponential_decay = -0.1 * y
        problem = {"x": y, "ode": exponential_decay}

        y0 = np.array([1])
        t_eval = np.linspace(0, 1, 100)
        solution = solver.integrate_casadi(problem, y0, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(-0.1 * solution.t))
        self.assertEqual(solution.termination, "final time")

    def test_integrate_failure(self):
        # Turn off warnings to ignore sqrt error
        warnings.simplefilter("ignore")

        y = casadi.SX.sym("y")
        sqrt_decay = -np.sqrt(y)

        y0 = np.array([1])
        t_eval = np.linspace(0, 3, 100)
        solver = pybamm.CasadiSolver(
            rtol=1e-8,
            atol=1e-8,
            method="idas",
            disable_internal_warnings=True,
            regularity_check=False,
        )
        problem = {"x": y, "ode": sqrt_decay}
        # Expect solver to fail when y goes negative
        with self.assertRaises(pybamm.SolverError):
            solver.integrate_casadi(problem, y0, t_eval)

        # Turn warnings back on
        warnings.simplefilter("default")

    def test_model_solver(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)
        # Solve
        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")
        t_eval = np.linspace(0, 1, 100)
        solution = solver.solve(model, t_eval)
        np.testing.assert_array_equal(solution.t, t_eval)
        np.testing.assert_allclose(solution.y[0], np.exp(0.1 * solution.t))

        # Test time
        self.assertGreater(
            solution.total_time, solution.solve_time + solution.set_up_time
        )

    def test_model_step(self):
        # Create model
        model = pybamm.BaseModel()
        domain = ["negative electrode", "separator", "positive electrode"]
        var = pybamm.Variable("var", domain=domain)
        model.rhs = {var: 0.1 * var}
        model.initial_conditions = {var: 1}
        # No need to set parameters; can use base discretisation (no spatial operators)

        # create discretisation
        mesh = get_mesh_for_testing()
        spatial_methods = {"macroscale": pybamm.FiniteVolume}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        solver = pybamm.CasadiSolver(rtol=1e-8, atol=1e-8, method="idas")

        # Step once
        dt = 0.1
        step_sol = solver.step(model, dt)
        np.testing.assert_array_equal(step_sol.t, [0, dt])
        np.testing.assert_allclose(step_sol.y[0], np.exp(0.1 * step_sol.t))

        # Step again (return 5 points)
        step_sol_2 = solver.step(model, dt, npts=5)
        np.testing.assert_array_equal(step_sol_2.t, np.linspace(dt, 2 * dt, 5))
        np.testing.assert_allclose(step_sol_2.y[0], np.exp(0.1 * step_sol_2.t))

        # append solutions
        step_sol.append(step_sol_2)

        # Check steps give same solution as solve
        t_eval = step_sol.t
        solution = solver.solve(model, t_eval)
        np.testing.assert_allclose(solution.y[0], step_sol.y[0])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
