#
# Shared methods and classes for testing
#
import pybamm
from scipy.sparse import eye


class SpatialMethodForTesting(pybamm.SpatialMethod):
    """Identity operators, no boundary conditions."""

    def __init__(self, options=None):
        super().__init__(options)

    def build(self, mesh):
        super().build(mesh)

    def gradient(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        gradient_matrix = pybamm.Matrix(eye(n))
        return gradient_matrix @ discretised_symbol

    def divergence(self, symbol, discretised_symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        divergence_matrix = pybamm.Matrix(eye(n))
        return divergence_matrix @ discretised_symbol

    def internal_neumann_condition(
        self, left_symbol_disc, right_symbol_disc, left_mesh, right_mesh
    ):
        return pybamm.Scalar(0)

    def mass_matrix(self, symbol, boundary_conditions):
        n = 0
        for domain in symbol.domain:
            n += self.mesh[domain][0].npts
        mass_matrix = pybamm.Matrix(eye(n))
        return mass_matrix


def get_mesh_for_testing(
    xpts=None, rpts=10, ypts=15, zpts=15, geometry=None, cc_submesh=None
):
    param = pybamm.ParameterValues(
        values={
            "Electrode width [m]": 0.4,
            "Electrode height [m]": 0.5,
            "Negative tab width [m]": 0.1,
            "Negative tab centre y-coordinate [m]": 0.1,
            "Negative tab centre z-coordinate [m]": 0.0,
            "Positive tab width [m]": 0.1,
            "Positive tab centre y-coordinate [m]": 0.3,
            "Positive tab centre z-coordinate [m]": 0.5,
            "Negative electrode thickness [m]": 0.3,
            "Separator thickness [m]": 0.3,
            "Positive electrode thickness [m]": 0.3,
        }
    )

    if geometry is None:
        geometry = pybamm.Geometry("1D macro", "1D micro")
    param.process_geometry(geometry)

    submesh_types = {
        "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        "current collector": pybamm.MeshGenerator(pybamm.SubMesh0D),
    }
    if cc_submesh:
        submesh_types["current collector"] = cc_submesh

    if xpts is None:
        xn_pts, xs_pts, xp_pts = 40, 25, 35
    else:
        xn_pts, xs_pts, xp_pts = xpts, xpts, xpts
    var = pybamm.standard_spatial_vars
    var_pts = {
        var.x_n: xn_pts,
        var.x_s: xs_pts,
        var.x_p: xp_pts,
        var.r_n: rpts,
        var.r_p: rpts,
        var.y: ypts,
        var.z: zpts,
    }

    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_p2d_mesh_for_testing(xpts=None, rpts=10):
    geometry = pybamm.Geometry("1D macro", "1+1D micro")
    return get_mesh_for_testing(xpts=xpts, rpts=rpts, geometry=geometry)


def get_1p1d_mesh_for_testing(
    xpts=None, zpts=15, cc_submesh=pybamm.MeshGenerator(pybamm.Uniform1DSubMesh)
):
    geometry = pybamm.Geometry("1+1D macro")
    return get_mesh_for_testing(
        xpts=xpts, zpts=zpts, geometry=geometry, cc_submesh=cc_submesh
    )


def get_2p1d_mesh_for_testing(
    xpts=None,
    ypts=15,
    zpts=15,
    cc_submesh=pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
):
    geometry = pybamm.Geometry("2+1D macro")
    return get_mesh_for_testing(
        xpts=xpts, zpts=zpts, geometry=geometry, cc_submesh=cc_submesh
    )


def get_unit_2p1D_mesh_for_testing(ypts=15, zpts=15):
    param = pybamm.ParameterValues(
        values={
            "Electrode width [m]": 1,
            "Electrode height [m]": 1,
            "Negative tab width [m]": 1,
            "Negative tab centre y-coordinate [m]": 0.5,
            "Negative tab centre z-coordinate [m]": 0,
            "Positive tab width [m]": 1,
            "Positive tab centre y-coordinate [m]": 0.5,
            "Positive tab centre z-coordinate [m]": 1,
            "Negative electrode thickness [m]": 0.3,
            "Separator thickness [m]": 0.3,
            "Positive electrode thickness [m]": 0.3,
        }
    )

    geometry = pybamm.Geometryxp1DMacro(cc_dimension=2)
    param.process_geometry(geometry)

    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 3, var.x_s: 3, var.x_p: 3, var.y: ypts, var.z: zpts}

    submesh_types = {
        "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        "current collector": pybamm.MeshGenerator(pybamm.ScikitUniform2DSubMesh),
    }

    return pybamm.Mesh(geometry, submesh_types, var_pts)


def get_discretisation_for_testing(
    xpts=None, rpts=10, mesh=None, cc_method=SpatialMethodForTesting
):
    if mesh is None:
        mesh = get_mesh_for_testing(xpts=xpts, rpts=rpts)
    spatial_methods = {
        "macroscale": SpatialMethodForTesting(),
        "negative particle": SpatialMethodForTesting(),
        "positive particle": SpatialMethodForTesting(),
        "current collector": cc_method(),
    }
    return pybamm.Discretisation(mesh, spatial_methods)


def get_p2d_discretisation_for_testing(xpts=None, rpts=10):
    return get_discretisation_for_testing(mesh=get_p2d_mesh_for_testing(xpts, rpts))


def get_1p1d_discretisation_for_testing(xpts=None, zpts=15):
    return get_discretisation_for_testing(mesh=get_1p1d_mesh_for_testing(xpts, zpts))


def get_2p1d_discretisation_for_testing(xpts=None, ypts=15, zpts=15):
    return get_discretisation_for_testing(
        mesh=get_2p1d_mesh_for_testing(xpts, ypts, zpts),
        cc_method=pybamm.ScikitFiniteElement,
    )
