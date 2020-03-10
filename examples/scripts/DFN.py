#
# Example showing how to load and solve the DFN
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")


# load model
model = pybamm.lithium_ion.BasicDFN()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
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
t_eval = np.linspace(0, 3600, 100)
solver = pybamm.CasadiSolver()
solver.rtol = 1e-3
solver.atol = 1e-6
solution = solver.solve(model, t_eval)

# # export
# import ipdb 
# ipdb.set_trace()
# c_s_n = pybamm.ProcessedVariable(model.variables["Negative particle concentration [mol.m-3]"], solution.t, solution.y, mesh,)
# from scipy.io import savemat
# savemat("file.mat",{"conc": c_s_n.entries})


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
