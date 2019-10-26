#
# Simulation class for a battery model
#
import pybamm
import numpy as np


class Simulation(object):
    """
    The simulation class for a battery model.

    Parameters
    ---------
    model : :class:`pybamm.BaseModel`
       The model to be simulated.
    parameter_values : :class:`pybamm.ParameterValues`, optional
       The parameters to be used for the simulation. Defaults to model default.
    geometry : :class:`pybamm.Geometry`, optional
       The geometry for the simulation. Defaults to model default.
    submesh_types : dict of :class:`pybamm.MeshGenerator`, optional
       The mesh types to be used for the simulation. Defaults to model default.
    var_pts : dict of int, optional
       The number of grid points in each subdomain. Defaults to model default.
    spatial_method : dict of :class:`pybamm.SpatialMethod`, optional
       The spatial discretisation for each subdomain. Defaults to model default.
    solver : :class:`pybamm.solver.Solver`, optional
       The algorithm for solving the model. Defaults to model default.
    name : string, optional
       The simulation name. Default is "Simulation for {model name}".

    Examples
    --------
    >>> import pybamm
    >>> model = pybamm.lithium_ion.DFN()
    >>> sim = pybamm.Simulation(model)
    >>> sim.run(t_eval=np.linspace(0, 1))

    """

    def __init__(
        self,
        model,
        parameter_values=None,
        geometry=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        name=None,
    ):
        # Read defaults from model
        parameter_values = parameter_values or model.default_parameter_values
        geometry = geometry or model.default_geometry
        submesh_types = submesh_types or model.default_submesh_types
        var_pts = var_pts or model.default_var_pts
        spatial_methods = spatial_methods or model.default_spatial_methods
        solver = solver or model.default_solver
        name = name or "Simulation for {}".format(model.name)

        # Assign attributes
        self.model = model
        self.parameter_values = parameter_values
        parameter_values.process_geometry(geometry)
        self.geometry = geometry
        self.mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
        self.discretisation = pybamm.Discretisation(self.mesh, spatial_methods)
        self.solver = solver
        self.name = name

    def __str__(self):
        return self.name

    def run(self, t_eval, plot=False, output_variables=None):
        self.parameter_values.process_model(self.model)
        self.discretisation.process_model(self.model)
        self.solution = self.solver.solve(self.model, t_eval)
        if plot is True:
            self.plot(output_variables)

    def save(self, filename):
        "Save model results to file"
        raise NotImplementedError

    def load(self, filename):
        "Load model results to file"
        raise NotImplementedError

    def plot(self, output_variables=None):
        "Quick plot of simulation results"
        plot = pybamm.QuickPlot(
            [self.model], self.mesh, [self.solution], output_variables
        )
        plot.dynamic_plot()
