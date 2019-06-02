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

    def test_well_posed_with_convection(self):
        model = pybamm.lead_acid.LOQS({"convection": True})
        model.check_well_posedness()

    def test_default_geometry(self):
        model = pybamm.lead_acid.LOQS()
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_default_spatial_methods(self):
        model = pybamm.lead_acid.LOQS()
        self.assertIsInstance(model.default_spatial_methods, dict)
        self.assertTrue("negative particle" not in model.default_geometry)

    def test_incompatible_options(self):
        options = {"bc_options": {"dimensionality": 1}}
        with self.assertRaises(pybamm.ModelError):
            pybamm.lead_acid.LOQS(options)


class TestLeadAcidLOQSCapacitance(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"capacitance": "algebraic"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_1plus1D(self):
        options = {"capacitance": "differential", "bc_options": {"dimensionality": 1}}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_default_solver(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsOdeSolver)
        options = {"capacitance": "algebraic"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_solver, pybamm.ScikitsDaeSolver)

    def test_default_geometry(self):
        options = {"capacitance": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertNotIn("current collector", model.default_geometry)
        options["bc_options"] = {"dimensionality": 1}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIn("current collector", model.default_geometry)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
