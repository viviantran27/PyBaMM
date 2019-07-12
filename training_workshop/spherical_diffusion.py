# Modelling linear diffusion in a sphere using PyBaMM

import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("DEBUG")

# 1. Initialise model ------------------------------------------------------------------
model = pybamm.BaseModel()

# 2. Define parameters and variables ---------------------------------------------------
R = pybamm.Parameter("Particle radius [m]")
D = pybamm.Parameter("Diffusion coefficient [m2.s-1]")
j = pybamm.Parameter("Molar ionic flux [mol.m-2.s-1]")
c0 = pybamm.Parameter("Initial concentration [mol.m-3]")

c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")

# 3. State governing equations ---------------------------------------------------------
N = -D * pybamm.grad(c)  # flux
dcdt = -pybamm.div(N)

model.rhs = {c: dcdt}  # add equation to rhs dictionary

# 4. State boundary conditions ---------------------------------------------------------
lbc = pybamm.Scalar(0)
rbc = -j / D
model.boundary_conditions = {c: {"left": (lbc, "Dirichlet"), "right": (rbc, "Neumann")}}

# 5. State initial conditions ----------------------------------------------------------
model.initial_conditions = {c: c0}

# 6. State output variables ------------------------------------------------------------
model.variables = {"Concentration [mol.m-3]": c, "Flux [mol.m-2.s-1]": N}

"--------------------------------------------------------------------------------------"
"Using the model"

# define geometry
r = pybamm.SpatialVariable(
    "r", domain=["negative particle"], coord_sys="spherical polar"
)
geometry = {"negative particle": {"primary": {r: {"min": pybamm.Scalar(0), "max": R}}}}

# parameter values
param = pybamm.ParameterValues(
    {
        "Particle radius [m]": 10e-6,
        "Diffusion coefficient [m2.s-1]": 3.9e-14,
        "Molar ionic flux [mol.m-2.s-1]": 6e-7,
        "Initial concentration [mol.m-3]": 2e5,
    }
)


# process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# mesh and discretise
submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
var_pts = {r: 200}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

spatial_methods = {"negative particle": pybamm.FiniteVolume}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 10, 100)
solution = solver.solve(model, t)

# Extract output variables
c_out = pybamm.ProcessedVariable(
    model.variables["Concentration [mol.m-3]"], solution.t, solution.y, mesh
)

# plot
plt.plot(solution.t, c_out(solution.t, r=10e-6))
plt.xlabel("Time [s]")
plt.ylabel("Concentration [mol.m-3]")
plt.show()
