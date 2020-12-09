#
# Incorporates thermal the model with the lead-acid model 
#

import pybamm
import numpy as np

# load model
pybamm.set_logging_level("INFO")

options = {"thermal": "x-full"}
full_thermal_model = pybamm.lead_acid.Full(options)

models = [full_thermal_model]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update({"Edge heat transfer coefficient [W.m-2.K-1]": 1})

for model in models:
    param.process_model(model)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 10, var.x_s: 10, var.x_p: 10, var.r_n: 10, var.r_p: 10}

# discretise models
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# solve model
solutions = [None] * len(models)
t_eval = np.linspace(0, 3600, 100)
for i, model in enumerate(models):
    solver = pybamm.CasadiSolver(atol=1e-8, rtol=1e-8)
    solution = solver.solve(model, t_eval)
    solutions[i] = solution

# plot
output_variables = [
    "Terminal voltage [V]",
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
]
labels = ["Full thermal model"]
plot = pybamm.QuickPlot(solutions, output_variables, labels)
plot.dynamic_plot()