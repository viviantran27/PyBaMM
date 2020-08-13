#
# Pulse test lead-acid battery (Broken)
#

import pybamm
import numpy as np

# load model
pybamm.set_logging_level("INFO")

class ExternalCircuitResistanceFunction:

    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return V / I - pybamm.FunctionParameter("Resistance [ohm]", {"Time [s]": pybamm.t})

def pulse_test(pulse_time, rest_time, pulse_current):
    def current(t):
        floor = pybamm.Function(np.floor, t/(pulse_time + rest_time))
        mod_t = t-(pulse_time + rest_time)*floor
        pulse_signal = mod_t < pulse_time
        return pulse_signal * pulse_current
    return current

options = {
    "thermal": "x-full",
    "operating mode": "current", 
}
full_thermal_model = pybamm.lead_acid.Full(options)

models = [full_thermal_model]

# load parameter values and process models and geometry
param = models[0].default_parameter_values
param.update({
    "Edge heat transfer coefficient [W.m-2.K-1]": 10,
    "Current function [A]": pulse_test(1*60, 1*60, 17*2),
    },
    check_already_exists=False
)

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
    "Electrolyte concentration [Molar]",
    "Current [A]",
]
labels = ["Full thermal model"]
plot = pybamm.QuickPlot(solutions, output_variables, labels)
plot.dynamic_plot()
