#
# Tests for the Manufactured Solution class
#
import pybamm
from tests import get_discretisation_for_testing

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
        x_n = pybamm.SpatialVariable("x", domain=a.domain)
        x = pybamm.SpatialVariable("x", domain=b.domain)
        r_n = pybamm.SpatialVariable("r", domain=c.domain)
        man_vars = {
            a.id: pybamm.Function(anp.cos, x_n),
            b.id: (x + 5) ** 2,
            c.id: pybamm.Function(anp.exp, 3 * r_n),
        }

        # Create manufactured solution class, and discretisation class for testing
        ms = pybamm.ManufacturedSolution()
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        # Process variable
        a_proc = ms.process_symbol(a, man_vars)
        self.assertEqual(a_proc, man_vars[a.id])
        b_proc = ms.process_symbol(b, man_vars)
        self.assertEqual(b_proc, man_vars[b.id])
        c_proc = ms.process_symbol(c, man_vars)
        self.assertEqual(c_proc, man_vars[c.id])

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
        ]:
            eqn_proc = ms.process_symbol(eqn, man_vars)
            eqn_proc_disc = disc.process_symbol(eqn_proc)
            np.testing.assert_almost_equal(
                eqn_proc_disc.evaluate(), expected, decimal=14
            )

    def test_manufacture_model(self):
        ms = pybamm.ManufacturedSolution()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
