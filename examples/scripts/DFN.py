#
# Example showing how to load and solve the DFN
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")
class ExternalCircuitFunction:
    num_switches = 0

    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return V / I - pybamm.FunctionParameter("Function", pybamm.t)

options = {"operating mode": ExternalCircuitFunction()}
model = pybamm.lithium_ion.DFN(options)
model.events = {}

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.update({"Function": .75}, check_already_exists=False)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.06, 100)
solver = model.default_solver
solver.rtol = 1e-8
solver.atol = 1e-7
solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(solution,output_variables = [
                    "Negative particle surface concentration",
                    "Electrolyte concentration",
                    "Positive particle surface concentration",
                    "Current [A]",
                    "Negative electrode potential [V]",
                    "Electrolyte potential [V]",
                    "Interfacial current density",
                    "Terminal voltage [V]",
                ])
plot.dynamic_plot()
