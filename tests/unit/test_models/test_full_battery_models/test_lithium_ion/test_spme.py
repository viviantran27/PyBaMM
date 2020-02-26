#
# Tests for the lithium-ion SPMe model
#
import pybamm
import unittest


class TestSPMe(unittest.TestCase):
    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIsInstance(model.default_geometry, pybamm.Geometry)
        self.assertTrue("negative particle" in model.default_geometry)

        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIn("current collector", model.default_geometry)

        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.SPMe(options)
        self.assertIn("current collector", model.default_geometry)

    def test_well_posed_2plus1D(self):
        options = {"current collector": "potential pair", "dimensionality": 1}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {"current collector": "potential pair", "dimensionality": 2}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {"bc_options": {"dimensionality": 5}}
        with self.assertRaises(pybamm.OptionError):
            model = pybamm.lithium_ion.SPMe(options)

    def test_x_full_thermal_model_no_current_collector(self):
        options = {"thermal": "x-full"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        # Not implemented with current collectors
        options = {"thermal": "x-full", "thermal current collector": True}
        with self.assertRaises(NotImplementedError):
            model = pybamm.lithium_ion.SPMe(options)

    def test_x_full_Nplus1D_not_implemented(self):
        # 1plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.SPMe(options)
        # 2plus1D
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-full",
        }
        with self.assertRaises(NotImplementedError):
            pybamm.lithium_ion.SPMe(options)

    def test_x_lumped_thermal_model_no_Current_collector(self):
        options = {"thermal": "x-lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        # xyz-lumped returns the same as x-lumped
        options = {"thermal": "xyz-lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_x_lumped_thermal_model_0D_current_collector(self):
        options = {"thermal": "x-lumped", "thermal current collector": True}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        # xyz-lumped returns the same as x-lumped
        options = {"thermal": "xyz-lumped", "thermal current collector": True}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_xyz_lumped_thermal_1D_current_collector(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "xyz-lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_xyz_lumped_thermal_2D_current_collector(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "xyz-lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_x_lumped_thermal_1D_current_collector(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_x_lumped_thermal_2D_current_collector(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "x-lumped",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_x_lumped_thermal_set_temperature_1D(self):
        options = {
            "current collector": "potential pair",
            "dimensionality": 1,
            "thermal": "set external temperature",
        }
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

        options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "thermal": "set external temperature",
        }
        with self.assertRaises(NotImplementedError):
            model = pybamm.lithium_ion.SPMe(options)

    def test_particle_fast_diffusion(self):
        options = {"particle": "fast diffusion"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_surface_form_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()

    def test_surface_form_algebraic(self):
        options = {"surface form": "algebraic"}
        model = pybamm.lithium_ion.SPMe(options)
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
