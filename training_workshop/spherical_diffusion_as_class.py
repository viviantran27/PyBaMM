# Modelling linear diffusion in a sphere using PyBaMM

import pybamm
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialise model ------------------------------------------------------------------
param = pybamm.my_parameters
model = pybamm.MySphericalDiffusion(param)

"--------------------------------------------------------------------------------------"
"Using the model"

# define geometry
r = pybamm.SpatialVariable(
    "r", domain=["negative particle"], coord_sys="spherical polar"
)
geometry = {
    "negative particle": {"primary": {r: {"min": pybamm.Scalar(0), "max": param.R}}}
}

# parameter values
param = pybamm.ParameterValues(
    {
        "Particle radius [m]": 10e-6,
        "Diffusion coefficient [m2.s-1]": 3.9e-14,
        "Interfacial current density [A.m-2]": 1.4,
        "Faraday constant []": 96485,
        "Initial concentration [mol.m-3]": 2.5e4,
    }
)


# process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# mesh and discretise
submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
var_pts = {r: 20}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"negative particle": pybamm.FiniteVolume}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 900)
solution = solver.solve(model, t)

# quick plot (won't work?)
# output_variables = ["Concentration [mol.m-3]", "Surface concentration [mol.m-3]", "Flux [mol.m-2.s-1]"]
# plot = pybamm.QuickPlot(model, mesh, solution, output_variables)
# plot.dynamic_plot()

# Extract output variables
c_surf = pybamm.ProcessedVariable(
    model.variables["Surface concentration [mol.m-3]"], solution.t, solution.y, mesh
)

# plot
plt.plot(solution.t, c_surf(solution.t))
plt.xlabel("Time [s]")
plt.ylabel("Surface concentration [mol.m-3]")
plt.show()
