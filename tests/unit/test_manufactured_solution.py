#
# Tests for the Manufactured Solution class
#
import pybamm
from tests import get_discretisation_for_testing, get_mesh_for_testing

import autograd.numpy as anp
import numpy as np
import unittest


class TestManufacturedSolution(unittest.TestCase):
    def test_manufacture_variable(self):
        ms = pybamm.ManufacturedSolution()
        manufactured_var = ms.create_manufactured_variable(domain=["negative particle"])

        # Discretise to test
        disc = get_discretisation_for_testing()
        mesh = disc.mesh
        man_var_disc = disc.process_symbol(manufactured_var)
        self.assertEqual(
            man_var_disc.evaluate(t=0).shape, (mesh["negative particle"][0].npts,)
        )

    def test_process_symbol(self):
        # Create known variables
        a = pybamm.Variable("a", domain=["negative electrode"])
        b = pybamm.Variable(
            "b", domain=["negative electrode", "separator", "positive electrode"]
        )
        c = pybamm.Variable("c", domain=["negative particle"])
        d = pybamm.Variable("d", domain=["negative electrode"])
        x_n = pybamm.SpatialVariable("x", domain=a.domain)
        x = pybamm.SpatialVariable("x", domain=b.domain)
        r_n = pybamm.SpatialVariable("r", domain=c.domain)
        manufactured_variables = {
            a.id: pybamm.Function(anp.cos, x_n),
            b.id: (x + 5) ** 2,
            c.id: pybamm.Function(anp.exp, 3 * r_n),
            d.id: x,
        }

        # Create manufactured solution class, and discretisation class for testing
        ms = pybamm.ManufacturedSolution()
        ms.manufactured_variables = manufactured_variables
        disc = get_discretisation_for_testing()

        # Process variable
        a_proc = ms.process_symbol(a)
        self.assertEqual(a_proc, manufactured_variables[a.id])
        b_proc = ms.process_symbol(b)
        self.assertEqual(b_proc, manufactured_variables[b.id])
        c_proc = ms.process_symbol(c)
        self.assertEqual(c_proc, manufactured_variables[c.id])
        d_proc = ms.process_symbol(d)
        self.assertEqual(d_proc, manufactured_variables[d.id])

        # Process equations
        x_n_eval = disc.process_symbol(x_n).evaluate()
        x_eval = disc.process_symbol(x).evaluate()
        r_n_eval = disc.process_symbol(r_n).evaluate()
        # Discretise to test
        for eqn, expected in [
            (pybamm.grad(a), -np.sin(x_n_eval)),
            (pybamm.div(a), -np.sin(x_n_eval)),
            (3 * pybamm.grad(2 * a) + 2, -6 * np.sin(x_n_eval) + 2),
            (pybamm.grad(b), 2 * (x_eval + 5)),
            (pybamm.div(3 * (b - 2)), 6 * (x_eval + 5)),
            (pybamm.grad(c), 3 * np.exp(3 * r_n_eval)),
            (
                pybamm.div(c),
                1
                / (r_n_eval ** 2)
                * ((2 * r_n_eval + 3 * r_n_eval ** 2) * np.exp(3 * r_n_eval)),
            ),
            (pybamm.grad(d), 1),
            (pybamm.div(pybamm.grad(d)), 0),
            (pybamm.div(pybamm.Vector(x_eval) * pybamm.grad(d)), 0),
        ]:
            eqn_proc = ms.process_symbol(eqn)
            eqn_proc_disc = disc.process_symbol(eqn_proc)
            np.testing.assert_almost_equal(
                eqn_proc_disc.evaluate(), expected, decimal=14
            )

    @unittest.skip("")
    def test_manufacture_model(self):
        # Create known variables
        a = pybamm.Variable("a", domain=["negative electrode"])
        b = pybamm.Variable(
            "b", domain=["negative electrode", "separator", "positive electrode"]
        )
        c = pybamm.Variable("c", domain=["negative particle"])
        x_n = pybamm.SpatialVariable("x", domain=a.domain)
        x = pybamm.SpatialVariable("x", domain=b.domain)
        r_n = pybamm.SpatialVariable("r", domain=c.domain)
        manufactured_variables = {
            a.id: pybamm.Function(anp.cos, x_n),
            b.id: (x + 5) ** 2,
            c.id: pybamm.Function(anp.sin, 3 * r_n),
        }

        # Create manufactured solution class, and discretisation class for testing
        ms = pybamm.ManufacturedSolution()
        mesh = get_mesh_for_testing(200)
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
        }
        disc = pybamm.Discretisation(mesh, spatial_methods)

        # Create and process model
        model = pybamm.BaseModel()
        flux_a = a * pybamm.grad(a)
        flux_c = pybamm.Function(anp.cosh, c) * pybamm.grad(c)
        model.algebraic = {a: pybamm.div(flux_a) + 5, b: -3 * b, c: pybamm.div(flux_c)}
        # initial and boundary conditions will be overwritten
        model.initial_conditions = {a: 0, b: 0, c: 0}
        model.boundary_conditions = {
            flux_a: {"left": 0, "right": 0},
            c: {"left": 0, "right": 0},
        }
        ms.process_model(model, manufactured_variables)

        # Set up tests
        x_n_eval = disc.process_symbol(x_n).evaluate()
        x_eval = disc.process_symbol(x).evaluate()
        r_n_eval = disc.process_symbol(r_n).evaluate()
        y_a = np.cos(x_n_eval)
        y_b = (x_eval + 5) ** 2
        y_c = np.sin(3 * r_n_eval)
        y = np.concatenate([y_a, y_b, y_c])
        # Discretise model and check answer
        disc.process_model(model)

        # Check initial and boundary conditions
        np.testing.assert_array_equal(model.initial_conditions[a].evaluate(y=y), y_a)
        np.testing.assert_array_equal(model.initial_conditions[b].evaluate(y=y), y_b)
        np.testing.assert_array_equal(model.initial_conditions[c].evaluate(y=y), y_c)
        np.testing.assert_almost_equal(
            model.boundary_conditions[flux_a]["left"].evaluate(y=y), 0
        )
        # note l_n = 1/3 in mesh_for_testing
        np.testing.assert_almost_equal(
            model.boundary_conditions[flux_a]["right"].evaluate(y=y),
            -np.cos(1 / 3) * np.sin(1 / 3),
            decimal=6,
        )
        np.testing.assert_almost_equal(
            model.boundary_conditions[c]["left"].evaluate(y=y), 0, decimal=4
        )
        np.testing.assert_almost_equal(
            model.boundary_conditions[c]["right"].evaluate(y=y), np.sin(3), decimal=1
        )
        # Check that algebraic equations with the right y evaluate to zero
        np.testing.assert_almost_equal(model.algebraic[a].evaluate(y=y), 0, decimal=3)
        np.testing.assert_almost_equal(model.algebraic[b].evaluate(y=y), 0, decimal=3)
        np.testing.assert_almost_equal(model.algebraic[c].evaluate(y=y), 0, decimal=0)

        # todo: test time derivative


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
