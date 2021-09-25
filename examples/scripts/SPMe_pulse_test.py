#
# Example showing how to load and solve the SPMe
#

import pybamm
import numpy as np
from shutil import copy

pybamm.set_logging_level("INFO")

def pulse_test(pulse_time, rest_time, pulse_current):
    def current(t):
        floor = pybamm.Function(np.floor, t/(pulse_time + rest_time))
        mod_t = t-(pulse_time + rest_time)*floor
        pulse_signal = mod_t < pulse_time
        return pulse_signal * pulse_current
    return current

# load model
filename = "DFN_pulse_17C.csv"
C = 1
options = {
    # "current collector": "potential pair",
    # "thermal": "x-lumped",
    # "side reactions": "decomposition",
    "operating mode": "current", }
model = pybamm.lithium_ion.SPMe(options)

# load parameter values and process model and geometry
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
soc_0 = 1
param.update(
    {
    "Current function [A]": pulse_test(0.25*60, 5*60, C*4.5),
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
t_eval = np.linspace(0, 3600*1, 10000)
solution = sim.solve(solver=pybamm.CasadiSolver(mode="safe", dt_max= 0.001, extra_options_setup={"max_num_steps": 1000}), t_eval=t_eval)

solution.save_data(
    filename,
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "Volume-averaged cell temperature [K]",
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
]

sim.plot(output_variables)
