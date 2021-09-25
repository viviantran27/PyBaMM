#
# Example showing how to load and solve the SPMe
#

import pybamm
import numpy as np
from shutil import copy

pybamm.set_logging_level("INFO")

# load model
filename = "SPMe_10C_CC.csv"
C = 10
options = {
    # "current collector": "potential pair",
    # "thermal": "x-lumped",
    # "side reactions": "decomposition",
    "operating mode": "current", }
model = pybamm.lithium_ion.SPMe(options)

param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
soc_0 = 1
# add variable to confirm actual resistance is constant
V = model.variables["Terminal voltage [V]"]
I = model.variables["Current [A]"]
# R_tab = pybamm.Parameter("Tabbing resistance [Ohm]")
# R_ext = pybamm.Parameter("External resistance [Ohm]")


model.variables.update({
    # "Terminal voltage [V]": V - I*R_tab,
    "Actual resistance [Ohm]":V/I,
    }
)

param.update(
    {
    "Current function [A]": 4.6*C,
    "Nominal cell capacity [A.h]": 4.6, 
    "Typical current [A]": 4.6,
    "Ambient temperature [K]":297.2909077,
    "Initial temperature [K]": 297.2909077,
    "Initial concentration in negative electrode [mol.m-3]":(soc_0*(0.87-0.0017)+0.0017)*28746, #x0 (0.0017) * Csmax_n(28746)
    "Initial concentration in positive electrode [mol.m-3]":(0.8907-soc_0*(0.8907-0.03))*35380, #y0 (0.8907) * Csmax_p(35380)
    "Negative electrode thickness [m]":62E-06 * 4.6/5, 
    "Positive electrode thickness [m]":67E-06 * 4.6/5,
    },
    check_already_exists=False,
)


# solve model
sim = pybamm.Simulation(model, parameter_values=param)
t_eval = np.linspace(0, 3600/C, 10000)
solution = sim.solve(solver=pybamm.CasadiSolver(mode="safe", dt_max= 0.001, extra_options_setup={"max_num_steps": 1000}), t_eval=t_eval)

solution.save_data(
    filename,
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "Volume-averaged cell temperature [K]",
        "Actual resistance [Ohm]",

    ],
    to_format="csv",
)

src = "C:/Users/Vivian/Documents/PyBaMM/" + filename 
dst = "C:/Users/Vivian/Box/Research/ESC modeling/dcr/" + filename 
copy(src, dst)

output_variables =[
    "Electrolyte concentration",
    "Electrolyte potential [V]",
    "Negative electrode potential [V]",
    "Current [A]",
    "Interfacial current density",
    "X-averaged cell temperature [K]",
    "Terminal voltage [V]",
    "X-averaged positive particle concentration",
    "X-averaged negative particle concentration",
    "Actual resistance [Ohm]",
]

sim.plot(output_variables)
