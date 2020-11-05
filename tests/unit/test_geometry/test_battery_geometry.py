#
# Tests for the base model class
#
import pybamm
import unittest


class TestBatteryGeometry(unittest.TestCase):
    def test_geometry_keys(self):
        for cc_dimension in [0, 1, 2]:
            geometry = pybamm.battery_geometry(current_collector_dimension=cc_dimension)
            for domain_geoms in geometry.values():
                all(
                    self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                    for spatial_var in domain_geoms.keys()
                )

    def test_geometry(self):
        var = pybamm.standard_spatial_vars
        geo = pybamm.geometric_parameters
        for cc_dimension in [0, 1, 2]:
            geometry = pybamm.battery_geometry(current_collector_dimension=cc_dimension)
            self.assertIsInstance(geometry, pybamm.Geometry)
            self.assertIn("negative electrode", geometry)
            self.assertIn("negative particle", geometry)
            self.assertEqual(geometry["negative electrode"][var.x_n]["min"], 0)
            self.assertEqual(
                geometry["negative electrode"][var.x_n]["max"].id, geo.l_n.id
            )
            if cc_dimension == 1:
                self.assertIn("tabs", geometry["current collector"])

        geometry = pybamm.battery_geometry(include_particles=False)
        self.assertNotIn("negative particle", geometry)

    def test_geometry_error(self):
        with self.assertRaisesRegex(pybamm.GeometryError, "Invalid current"):
            pybamm.battery_geometry(current_collector_dimension=4)


class TestReadParameters(unittest.TestCase):
    # This is the most complicated geometry and should test the parameters are
    # all returned for the deepest dict
    def test_read_parameters(self):
        geo = pybamm.geometric_parameters
        L_n = geo.L_n
        L_s = geo.L_s
        L_p = geo.L_p
        L_y = geo.L_y
        L_z = geo.L_z
        tab_n_y = geo.Centre_y_tab_n
        tab_n_z = geo.Centre_z_tab_n
        L_tab_n = geo.L_tab_n
        tab_p_y = geo.Centre_y_tab_p
        tab_p_z = geo.Centre_z_tab_p
        L_tab_p = geo.L_tab_p

        geometry = pybamm.battery_geometry(current_collector_dimension=2)

        self.assertEqual(
            set([x.name for x in geometry.parameters]),
            set(
                [
                    x.name
                    for x in [
                        L_n,
                        L_s,
                        L_p,
                        L_y,
                        L_z,
                        tab_n_y,
                        tab_n_z,
                        L_tab_n,
                        tab_p_y,
                        tab_p_z,
                        L_tab_p,
                    ]
                ]
            ),
        )
        self.assertTrue(
            all(isinstance(x, pybamm.Parameter) for x in geometry.parameters)
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
