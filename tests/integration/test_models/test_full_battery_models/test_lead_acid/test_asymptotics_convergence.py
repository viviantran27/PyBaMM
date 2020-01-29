#
# Tests for the asymptotic convergence of the simplified models
#
import pybamm

import numpy as np
import unittest


class TestAsymptoticConvergence(unittest.TestCase):
    def test_leading_order_convergence(self):
        """
        Check that the leading-order model solution converges linearly in C_e to the
        full model solution
        """
        # Create models
        leading_order_model = pybamm.lead_acid.LOQS()
        composite_model = pybamm.lead_acid.Composite()
        full_model = pybamm.lead_acid.Full()

        def current_function(t):
            return pybamm.InputParameter("Current")

        # Same parameters, same geometry
        parameter_values = full_model.default_parameter_values
        parameter_values["Current function [A]"] = current_function
        parameter_values.process_model(leading_order_model)
        parameter_values.process_model(composite_model)
        parameter_values.process_model(full_model)
        geometry = full_model.default_geometry
        parameter_values.process_geometry(geometry)

        # Discretise (same mesh, create different discretisations)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 3, var.x_s: 3, var.x_p: 3}
        mesh = pybamm.Mesh(geometry, full_model.default_submesh_types, var_pts)

        method_options = {"extrapolation": {"order": "linear", "use bcs": False}}
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume(method_options),
            "current collector": pybamm.ZeroDimensionalMethod(method_options),
        }
        loqs_disc = pybamm.Discretisation(mesh, spatial_methods)
        loqs_disc.process_model(leading_order_model)
        comp_disc = pybamm.Discretisation(mesh, spatial_methods)
        comp_disc.process_model(composite_model)
        full_disc = pybamm.Discretisation(mesh, spatial_methods)
        full_disc.process_model(full_model)

        def get_max_error(current):
            pybamm.logger.info("current = {}".format(current))
            # Solve, make sure times are the same and use tight tolerances
            t_eval = np.linspace(0, 0.6)
            solver_loqs = leading_order_model.default_solver
            solver_loqs.rtol = 1e-8
            solver_loqs.atol = 1e-8
            solution_loqs = solver_loqs.solve(
                leading_order_model, t_eval, inputs={"Current": current}
            )
            solver_comp = composite_model.default_solver
            solver_comp.rtol = 1e-8
            solver_comp.atol = 1e-8
            solution_comp = solver_comp.solve(
                composite_model, t_eval, inputs={"Current": current}
            )
            solver_full = full_model.default_solver
            solver_full.rtol = 1e-8
            solver_full.atol = 1e-8
            solution_full = solver_full.solve(
                full_model, t_eval, inputs={"Current": current}
            )

            # Post-process variables
            voltage_loqs = solution_loqs["Terminal voltage"]
            voltage_comp = solution_comp["Terminal voltage"]
            voltage_full = solution_full["Terminal voltage"]

            # Compare
            t_loqs = solution_loqs.t
            t_comp = solution_comp.t
            t_full = solution_full.t
            t = t_full[: np.min([len(t_loqs), len(t_comp), len(t_full)])]
            loqs_error = np.max(np.abs(voltage_loqs(t) - voltage_full(t)))
            comp_error = np.max(np.abs(voltage_comp(t) - voltage_full(t)))
            return (loqs_error, comp_error)

        # Get errors
        currents = 0.5 / (2 ** np.arange(3))
        errs = np.array([get_max_error(current) for current in currents])
        loqs_errs, comp_errs = [np.array(err) for err in zip(*errs)]
        # Get rates: expect linear convergence for loqs, quadratic for composite
        loqs_rates = np.log2(loqs_errs[:-1] / loqs_errs[1:])

        np.testing.assert_array_less(0.99 * np.ones_like(loqs_rates), loqs_rates)
        # Composite not converging as expected
        comp_rates = np.log2(comp_errs[:-1] / comp_errs[1:])
        np.testing.assert_array_less(0.99 * np.ones_like(comp_rates), comp_rates)
        # Check composite more accurate than loqs
        np.testing.assert_array_less(comp_errs, loqs_errs)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.set_logging_level("DEBUG")
    unittest.main()
