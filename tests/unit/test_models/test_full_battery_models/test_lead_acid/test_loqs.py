#
# Tests for the lead-acid LOQS model
#
import pybamm
import unittest


class TestLeadAcidLOQS(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

        # Test build after init
        model = pybamm.lead_acid.LOQS(build=False)
        model.build_model()
        model.check_well_posedness()

    def test_well_posed_with_convection(self):
        options = {"thermal": "isothermal", "convection": True}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_1plus1D(self):
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_2plus1D(self):
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 2,
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertNotIn("negative particle", model.default_geometry)

    def test_defaults_dimensions(self):
        model = pybamm.lead_acid.LOQS()
        self.assertIsInstance(model.default_spatial_methods, dict)
        self.assertNotIn("negative particle", model.default_geometry)
        self.assertTrue(
            isinstance(
                model.default_spatial_methods["current collector"],
                pybamm.ZeroDimensionalMethod,
            )
        )
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"].submesh_type,
                pybamm.SubMesh0D,
            )
        )
        model = pybamm.lead_acid.LOQS(
            {
                "surface form": "differential",
                "current collector": "potential pair",
                "dimensionality": 1,
            }
        )
        self.assertTrue(
            isinstance(
                model.default_spatial_methods["current collector"], pybamm.FiniteVolume
            )
        )
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"].submesh_type,
                pybamm.Uniform1DSubMesh,
            )
        )
        model = pybamm.lead_acid.LOQS(
            {
                "surface form": "differential",
                "current collector": "potential pair",
                "dimensionality": 2,
            }
        )
        self.assertTrue(
            isinstance(
                model.default_spatial_methods["current collector"],
                pybamm.ScikitFiniteElement,
            )
        )
        self.assertTrue(
            issubclass(
                model.default_submesh_types["current collector"].submesh_type,
                pybamm.ScikitUniform2DSubMesh,
            )
        )


class TestLeadAcidLOQSWithSideReactions(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic", "side reactions": ["oxygen"]}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_varying_surface_area(self):
        options = {
            "surface form": "differential",
            "side reactions": ["oxygen"],
            "interfacial surface area": "varying",
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_incompatible_options(self):
        options = {"side reactions": ["something"]}
        with self.assertRaises(pybamm.OptionError):
            pybamm.lead_acid.LOQS(options)


class TestLeadAcidLOQSSurfaceForm(unittest.TestCase):
    def test_well_posed_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_1plus1D(self):
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        self.assertIn("current collector", model.default_geometry)
        options.update({"current collector": "potential pair", "dimensionality": 1})
        model = pybamm.lead_acid.LOQS(options)
        self.assertIn("current collector", model.default_geometry)


class TestLeadAcidLOQSExternalCircuits(unittest.TestCase):
    def test_well_posed_voltage(self):
        options = {"operating mode": "voltage"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_power(self):
        options = {"operating mode": "power"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_function(self):
        def external_circuit_function(variables):
            I = variables["Current [A]"]
            V = variables["Terminal voltage [V]"]
            return V + I - pybamm.FunctionParameter("Function", pybamm.t)

        options = {"operating mode": external_circuit_function}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
