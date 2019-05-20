#
# Tests for the lead-acid LOQS model
#
import pybamm
import unittest


class TestLeadAcidLOQS(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.lead_acid.LOQS()
        model.check_well_posedness()
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
