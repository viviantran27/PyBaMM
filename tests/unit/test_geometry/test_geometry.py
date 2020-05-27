#
# Tests for the base model class
#
import pybamm
import unittest


class TestGeometry1DMacro(unittest.TestCase):
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry1DMacro()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.SpatialVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}
            }
        }
        geometry.update(custom_geometry)
        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self):
        geometry = pybamm.Geometry1DMacro()
        for prim_sec_vars in geometry.values():
            spatial_vars = prim_sec_vars["primary"]
            all(
                self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                for spatial_var in spatial_vars.keys()
            )


class TestGeometry1DMicro(unittest.TestCase):
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry1DMicro()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.SpatialVariable("x", whole_cell)
        custom_geometry = {}
        custom_geometry["negative electrode"] = {
            "primary": {x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}
        }
        geometry.update(custom_geometry)

        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self):
        geometry = pybamm.Geometry1DMicro()
        for prim_sec_vars in geometry.values():
            for spatial_vars in prim_sec_vars.values():
                all(
                    self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                    for spatial_var in spatial_vars.keys()
                )


class TestGeometry3DMacro(unittest.TestCase):
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry3DMacro()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.SpatialVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                "primary": {x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}
            }
        }

        geometry.update(custom_geometry)
        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self):
        geometry = pybamm.Geometry3DMacro()
        for prim_sec_vars in geometry.values():
            for spatial_vars in prim_sec_vars.values():
                all(
                    self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                    for spatial_var in spatial_vars.keys()
                )


class TestGeometry1p1DMacro(unittest.TestCase):
    def test_geometry_keys(self):
        geometry = pybamm.Geometryxp1DMacro(cc_dimension=1)
        for key, prim_sec_vars in geometry.items():
            self.assertIn("primary", prim_sec_vars.keys())
            if key != "current collector":
                self.assertIn("secondary", prim_sec_vars.keys())
                var = pybamm.standard_spatial_vars
                self.assertEqual(
                    list(prim_sec_vars["secondary"].keys())[0].id, var.z.id
                )
            for spatial_vars in prim_sec_vars.values():
                all(
                    self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                    for spatial_var in spatial_vars.keys()
                    if spatial_var not in ["negative", "positive"]
                )


class TestGeometry2p1DMacro(unittest.TestCase):
    def test_geometry_keys(self):
        geometry = pybamm.Geometryxp1DMacro(cc_dimension=2)
        for key, prim_sec_vars in geometry.items():
            self.assertIn("primary", prim_sec_vars.keys())
            if key != "current collector":
                self.assertIn("secondary", prim_sec_vars.keys())
                var = pybamm.standard_spatial_vars
                self.assertIn(var.y, prim_sec_vars["secondary"].keys())
                self.assertIn(var.z, prim_sec_vars["secondary"].keys())

            for spatial_vars in prim_sec_vars.values():
                all(
                    self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                    for spatial_var in spatial_vars.keys()
                    if spatial_var not in ["negative", "positive"]
                )

        def test_init_failure(self):
            with self.assertRaises(pybamm.GeometryError):
                pybamm.Geometryxp1DMacro(cc_dimension=3)


class TestGeometry1p1DMicro(unittest.TestCase):
    def test_geometry_keys(self):
        geometry = pybamm.Geometry1p1DMicro()
        for prim_sec_vars in geometry.values():
            for spatial_vars in prim_sec_vars.values():
                all(
                    self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                    for spatial_var in spatial_vars.keys()
                )


class TestGeometryxp0p1DMicro(unittest.TestCase):
    def test_geometry_keys(self):
        for cc_dimension in [1, 2]:
            geometry = pybamm.Geometryxp0p1DMicro(cc_dimension=cc_dimension)
            for prim_sec_vars in geometry.values():
                for spatial_vars in prim_sec_vars.values():
                    all(
                        self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                        for spatial_var in spatial_vars.keys()
                    )

    def test_init_failure(self):
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Geometryxp0p1DMicro(cc_dimension=3)


class TestGeometryxp1p1DMicro(unittest.TestCase):
    def test_geometry_keys(self):
        for cc_dimension in [1, 2]:
            geometry = pybamm.Geometryxp1p1DMicro(cc_dimension=cc_dimension)
            for prim_sec_vars in geometry.values():
                for spatial_vars in prim_sec_vars.values():
                    all(
                        self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                        for spatial_var in spatial_vars.keys()
                    )

    def test_init_failure(self):
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Geometryxp0p1DMicro(cc_dimension=3)


class TestGeometry(unittest.TestCase):
    def test_add_domain(self):
        L = pybamm.Parameter("L")
        zero = pybamm.Scalar(0)
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        geometry = pybamm.Geometry()

        geometry.add_domain("name_of_domain", {"primary": {x: {"min": zero, "max": L}}})

        geometry.add_domain("name_of_domain", {"tabs": {"negative": {"z_centre": L}}})

        # name must be a string
        with self.assertRaisesRegex(ValueError, "name must be a string"):
            geometry.add_domain(123, {"primary": {x: {"min": zero, "max": L}}})

        # keys of geometry must be either \"primary\" or \"secondary\"
        with self.assertRaisesRegex(ValueError, "primary.*secondary"):
            geometry.add_domain(
                "name_of_domain", {"primaryy": {x: {"min": zero, "max": L}}}
            )

        # inner dict of geometry must have pybamm.SpatialVariable as keys
        with self.assertRaisesRegex(ValueError, "pybamm\.SpatialVariable as keys"):
            geometry.add_domain(
                "name_of_domain", {"primary": {L: {"min": zero, "max": L}}}
            )

        # no minimum extents for variable
        with self.assertRaisesRegex(ValueError, "minimum"):
            geometry.add_domain("name_of_domain", {"primary": {x: {"max": L}}})

        # no maximum extents for variable
        with self.assertRaisesRegex(ValueError, "maximum"):
            geometry.add_domain("name_of_domain", {"primary": {x: {"min": zero}}})

        # tabs region must be \"negative\" or \"positive\"
        with self.assertRaisesRegex(ValueError, "negative.*positive"):
            geometry.add_domain(
                "name_of_domain", {"tabs": {"negativee": {"z_centre": L}}}
            )

        # tabs region params must be \"y_centre\", "\"z_centre\" or \"width\"
        with self.assertRaisesRegex(ValueError, "y_centre.*z_centre.*width"):
            geometry.add_domain(
                "name_of_domain", {"tabs": {"negative": {"z_centree": L}}}
            )

    def test_combine_geometries(self):
        geometry1Dmacro = pybamm.Geometry1DMacro()
        geometry1Dmicro = pybamm.Geometry1DMicro()
        geometry = pybamm.Geometry(geometry1Dmacro, geometry1Dmicro)
        self.assertEqual(
            set(geometry.keys()),
            set(
                [
                    "negative electrode",
                    "separator",
                    "positive electrode",
                    "negative particle",
                    "positive particle",
                    "current collector",
                ]
            ),
        )

        # update with custom geometry
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.SpatialVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                "primary": {x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}
            }
        }
        geometry = pybamm.Geometry(
            geometry1Dmacro, geometry1Dmicro, custom_geometry=custom_geometry
        )
        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_combine_geometries_3D(self):
        geometry3Dmacro = pybamm.Geometry3DMacro()
        geometry1Dmicro = pybamm.Geometry1DMicro()
        geometry = pybamm.Geometry(geometry3Dmacro, geometry1Dmicro)
        self.assertEqual(
            set(geometry.keys()),
            set(
                [
                    "negative electrode",
                    "separator",
                    "positive electrode",
                    "negative particle",
                    "positive particle",
                    "current collector",
                ]
            ),
        )

        with self.assertRaises(ValueError):
            geometry1Dmacro = pybamm.Geometry1DMacro()
            geometry = pybamm.Geometry(geometry3Dmacro, geometry1Dmacro)

    def test_combine_geometries_strings(self):
        geometry = pybamm.Geometry("1D macro", "1D micro")
        self.assertEqual(
            set(geometry.keys()),
            set(
                [
                    "negative electrode",
                    "separator",
                    "positive electrode",
                    "negative particle",
                    "positive particle",
                    "current collector",
                ]
            ),
        )
        geometry = pybamm.Geometry("3D macro", "1D micro")
        self.assertEqual(
            set(geometry.keys()),
            set(
                [
                    "negative electrode",
                    "separator",
                    "positive electrode",
                    "negative particle",
                    "positive particle",
                    "current collector",
                ]
            ),
        )


class TestGeometry1DCurrentCollector(unittest.TestCase):
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry1DCurrentCollector()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.SpatialVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                "primary": {x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}
            }
        }

        geometry.update(custom_geometry)
        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self):
        geometry = pybamm.Geometry1DCurrentCollector()
        for prim_sec_vars in geometry.values():
            for spatial_vars in prim_sec_vars.values():
                all(
                    self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                    for spatial_var in spatial_vars.keys()
                    if spatial_var not in ["negative", "positive"]
                )


class TestGeometry2DCurrentCollector(unittest.TestCase):
    def test_add_custom_geometry(self):
        geometry = pybamm.Geometry2DCurrentCollector()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        x = pybamm.SpatialVariable("x", whole_cell)
        custom_geometry = {
            "negative electrode": {
                "primary": {x: {"min": pybamm.Scalar(1), "max": pybamm.Scalar(2)}}
            }
        }

        geometry.update(custom_geometry)
        self.assertEqual(
            geometry["negative electrode"], custom_geometry["negative electrode"]
        )

    def test_geometry_keys(self):
        geometry = pybamm.Geometry2DCurrentCollector()
        for prim_sec_vars in geometry.values():
            for spatial_vars in prim_sec_vars.values():
                all(
                    self.assertIsInstance(spatial_var, pybamm.SpatialVariable)
                    for spatial_var in spatial_vars.keys()
                    if spatial_var not in ["negative", "positive"]
                )


class TestReadParameters(unittest.TestCase):
    # This is the most complicated geometry and should test the parameters are
    # all returned for the deepest dict
    def test_read_parameters(self):
        L_n = pybamm.geometric_parameters.L_n
        L_s = pybamm.geometric_parameters.L_s
        L_p = pybamm.geometric_parameters.L_p
        L_y = pybamm.geometric_parameters.L_y
        L_z = pybamm.geometric_parameters.L_z
        tab_n_y = pybamm.geometric_parameters.Centre_y_tab_n
        tab_n_z = pybamm.geometric_parameters.Centre_z_tab_n
        L_tab_n = pybamm.geometric_parameters.L_tab_n
        tab_p_y = pybamm.geometric_parameters.Centre_y_tab_p
        tab_p_z = pybamm.geometric_parameters.Centre_z_tab_p
        L_tab_p = pybamm.geometric_parameters.L_tab_p

        geometry = pybamm.Geometry("2+1D macro", "(2+1)+1D micro")

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
