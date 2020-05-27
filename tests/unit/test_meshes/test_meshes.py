#
# Test for the Finite Volume Mesh class
#
import pybamm
import numpy as np
import unittest


class TestMesh(unittest.TestCase):
    def test_mesh_creation_no_parameters(self):
        r = pybamm.SpatialVariable(
            "r", domain=["negative particle"], coord_sys="spherical polar"
        )

        geometry = {
            "negative particle": {
                "primary": {r: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}}
            }
        }

        submesh_types = {
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)
        }
        var_pts = {r: 20}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # create mesh
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative particle"][0].edges[0], 0)
        self.assertEqual(mesh["negative particle"][0].edges[-1], 1)

        # check number of edges and nodes
        self.assertEqual(len(mesh["negative particle"][0].nodes), var_pts[r])
        self.assertEqual(
            len(mesh["negative particle"][0].edges),
            len(mesh["negative particle"][0].nodes) + 1,
        )

    def test_submesh_creation(self):
        pybamm.SubMesh()

    def test_mesh_creation(self):
        param = pybamm.ParameterValues(
            values={
                "Negative electrode thickness [m]": 0.1,
                "Separator thickness [m]": 0.2,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        # check boundary locations
        self.assertEqual(mesh["negative electrode"][0].edges[0], 0)
        self.assertEqual(mesh["positive electrode"][0].edges[-1], 1)

        # check internal boundary locations
        self.assertEqual(
            mesh["negative electrode"][0].edges[-1], mesh["separator"][0].edges[0]
        )
        self.assertEqual(
            mesh["positive electrode"][0].edges[0], mesh["separator"][0].edges[-1]
        )
        for domain in mesh:
            if domain != "current collector":
                self.assertEqual(
                    len(mesh[domain][0].edges), len(mesh[domain][0].nodes) + 1
                )

    def test_init_failure(self):
        geometry = pybamm.Geometry1DMacro()
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }
        with self.assertRaisesRegex(KeyError, "Points not given"):
            pybamm.Mesh(geometry, submesh_types, {})

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12}
        geometry = pybamm.Geometry1p1DMicro()
        with self.assertRaisesRegex(KeyError, "Points not given"):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_mesh_sizes(self):
        param = pybamm.ParameterValues(
            values={
                "Negative electrode thickness [m]": 0.1,
                "Separator thickness [m]": 0.2,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        # provide mesh properties
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        var_id_pts = {var.id: pts for var, pts in var_pts.items()}

        self.assertEqual(mesh["negative electrode"][0].npts, var_id_pts[var.x_n.id])
        self.assertEqual(mesh["separator"][0].npts, var_id_pts[var.x_s.id])
        self.assertEqual(mesh["positive electrode"][0].npts, var_id_pts[var.x_p.id])

        self.assertEqual(
            len(mesh["negative electrode"][0].edges) - 1, var_id_pts[var.x_n.id]
        )
        self.assertEqual(len(mesh["separator"][0].edges) - 1, var_id_pts[var.x_s.id])
        self.assertEqual(
            len(mesh["positive electrode"][0].edges) - 1, var_id_pts[var.x_p.id]
        )

    def test_combine_submeshes(self):
        param = pybamm.ParameterValues(
            values={
                "Negative electrode thickness [m]": 0.1,
                "Separator thickness [m]": 0.2,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry("1D macro", "1D micro")
        param.process_geometry(geometry)

        # provide mesh properties
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        # create submesh
        submesh = mesh.combine_submeshes("negative electrode", "separator")
        self.assertEqual(submesh[0].edges[0], 0)
        self.assertEqual(submesh[0].edges[-1], mesh["separator"][0].edges[-1])
        np.testing.assert_almost_equal(
            submesh[0].nodes
            - np.concatenate(
                [mesh["negative electrode"][0].nodes, mesh["separator"][0].nodes]
            ),
            0,
        )
        np.testing.assert_almost_equal(submesh[0].internal_boundaries, [0.1 / 0.6])
        with self.assertRaises(pybamm.DomainError):
            mesh.combine_submeshes("negative electrode", "positive electrode")

        # test errors
        geometry = {
            "negative electrode": {
                "primary": {
                    var.x_n: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(0.5)}
                }
            },
            "negative particle": {
                "primary": {
                    var.r_n: {"min": pybamm.Scalar(0.5), "max": pybamm.Scalar(1)}
                }
            },
        }
        param.process_geometry(geometry)

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        with self.assertRaisesRegex(pybamm.DomainError, "trying"):
            mesh.combine_submeshes("negative electrode", "negative particle")

        with self.assertRaisesRegex(
            ValueError, "Submesh domains being combined cannot be empty"
        ):
            mesh.combine_submeshes()

    def test_ghost_cells(self):
        param = pybamm.ParameterValues(
            values={
                "Negative electrode thickness [m]": 0.1,
                "Separator thickness [m]": 0.2,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        # provide mesh properties
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"][0].edges[1],
            mesh["negative electrode"][0].edges[0],
        )
        np.testing.assert_array_equal(
            mesh["negative electrode_left ghost cell"][0].edges[0],
            -mesh["negative electrode"][0].edges[1],
        )
        np.testing.assert_array_equal(
            mesh["positive electrode_right ghost cell"][0].edges[0],
            mesh["positive electrode"][0].edges[-1],
        )

    def test_multiple_meshes(self):
        param = pybamm.ParameterValues(
            values={
                "Negative electrode thickness [m]": 0.1,
                "Separator thickness [m]": 0.2,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry("1+1D micro")
        param.process_geometry(geometry)

        # provide mesh properties

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_p: 10, var.r_n: 5, var.r_p: 6}
        submesh_types = {
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check types
        self.assertIsInstance(mesh["negative particle"], list)
        self.assertIsInstance(mesh["positive particle"], list)
        self.assertEqual(len(mesh["negative particle"]), 10)
        self.assertEqual(len(mesh["positive particle"]), 10)

        for i in range(10):
            self.assertIsInstance(mesh["negative particle"][i], pybamm.Uniform1DSubMesh)
            self.assertIsInstance(mesh["positive particle"][i], pybamm.Uniform1DSubMesh)
            self.assertEqual(mesh["negative particle"][i].npts, 5)
            self.assertEqual(mesh["positive particle"][i].npts, 6)

    def test_multiple_meshes_macro(self):
        param = pybamm.ParameterValues(
            values={
                "Negative electrode thickness [m]": 0.1,
                "Separator thickness [m]": 0.2,
                "Positive electrode thickness [m]": 0.3,
                "Electrode height [m]": 0.4,
                "Negative tab centre z-coordinate [m]": 0.0,
                "Positive tab centre z-coordinate [m]": 0.4,
            }
        )

        geometry = pybamm.Geometry("1+1D macro")
        param.process_geometry(geometry)
        # provide mesh properties

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 15, var.x_p: 20, var.z: 5}
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

        # check types
        self.assertIsInstance(mesh["negative electrode"], list)
        self.assertIsInstance(mesh["separator"], list)
        self.assertIsInstance(mesh["positive electrode"], list)
        self.assertEqual(len(mesh["negative electrode"]), 5)
        self.assertEqual(len(mesh["separator"]), 5)
        self.assertEqual(len(mesh["positive electrode"]), 5)

        for i in range(5):
            self.assertIsInstance(
                mesh["negative electrode"][i], pybamm.Uniform1DSubMesh
            )
            self.assertIsInstance(mesh["separator"][i], pybamm.Uniform1DSubMesh)
            self.assertIsInstance(
                mesh["positive electrode"][i], pybamm.Uniform1DSubMesh
            )
            self.assertEqual(mesh["negative electrode"][i].npts, 10)
            self.assertEqual(mesh["separator"][i].npts, 15)
            self.assertEqual(mesh["positive electrode"][i].npts, 20)

    def test_mesh_coord_sys(self):
        param = pybamm.ParameterValues(
            values={
                "Negative electrode thickness [m]": 0.1,
                "Separator thickness [m]": 0.2,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.Geometry1DMacro()
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 12, var.r_n: 5, var.r_p: 6}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        for submeshlist in mesh.values():
            for submesh in submeshlist:
                if not isinstance(submesh, pybamm.SubMesh0D):
                    self.assertTrue(submesh.coord_sys in pybamm.KNOWN_COORD_SYS)

    def test_unimplemented_meshes(self):
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.y: 10}
        geometry = {
            "negative electrode": {
                "primary": {var.x_n: {"min": 0, "max": 1}, var.y: {"min": 0, "max": 1}}
            }
        }
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)
        }
        with self.assertRaises(pybamm.GeometryError):
            pybamm.Mesh(geometry, submesh_types, var_pts)

    def test_1plus1D_tabs_left_right(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab centre z-coordinate [m]": 0.0,
                "Positive tab centre z-coordinate [m]": 0.5,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.Geometryxp1DMacro(cc_dimension=1)
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 7, var.x_p: 12, var.z: 24}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        # negative tab should be "left"
        self.assertEqual(mesh["current collector"][0].tabs["negative tab"], "left")

        # positive tab should be "right"
        self.assertEqual(mesh["current collector"][0].tabs["positive tab"], "right")

    def test_1plus1D_tabs_right_left(self):
        param = pybamm.ParameterValues(
            values={
                "Electrode width [m]": 0.4,
                "Electrode height [m]": 0.5,
                "Negative tab centre z-coordinate [m]": 0.5,
                "Positive tab centre z-coordinate [m]": 0.0,
                "Negative electrode thickness [m]": 0.3,
                "Separator thickness [m]": 0.3,
                "Positive electrode thickness [m]": 0.3,
            }
        )

        geometry = pybamm.Geometryxp1DMacro(cc_dimension=1)
        param.process_geometry(geometry)

        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 10, var.x_s: 7, var.x_p: 12, var.z: 24}

        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "current collector": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }

        mesh_type = pybamm.Mesh

        # create mesh
        mesh = mesh_type(geometry, submesh_types, var_pts)

        # negative tab should be "right"
        self.assertEqual(mesh["current collector"][0].tabs["negative tab"], "right")

        # positive tab should be "left"
        self.assertEqual(mesh["current collector"][0].tabs["positive tab"], "left")


class TestMeshGenerator(unittest.TestCase):
    def test_init_name(self):
        mesh_generator = pybamm.MeshGenerator(pybamm.SubMesh0D)
        self.assertEqual(mesh_generator.__repr__(), "Generator for SubMesh0D")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
