#
# Simulate drive cycle loaded from csv file
#
import pybamm
import pandas as pd
import os

os.chdir(pybamm.__path__[0] + "/..")

pybamm.set_logging_level("INFO")

# load model and update parameters so the input current is the drive cycle
model = pybamm.lithium_ion.SPMe() #{"thermal": "lumped"}
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
soc_0 = 0.20
param.update(
    {    
    "Cell capacity [A.h]": 4.6, 
    "Typical current [A]": 4.6,
    "Ambient temperature [K]":296.7,
    "Initial temperature [K]": 296.7,
    "Initial concentration in negative electrode [mol.m-3]":(soc_0*(0.75-0.0017)+0.0017)*28746, #x0 (0.0017) * Csmax_n(28746)
    "Initial concentration in positive electrode [mol.m-3]":(0.8907-soc_0*(0.8907-0.03))*35380, #y0 (0.8907) * Csmax_p(35380)
    }
)

# import drive cycle from file
drive_cycle = pd.read_csv(
    "pybamm/input/drive_cycles/thermal_pulse_test.csv", comment="#", header=None
).to_numpy()

# create interpolant
timescale = param.evaluate(model.timescale)
current_interpolant = pybamm.Interpolant(drive_cycle, timescale * pybamm.t)
param.update({"Current function [A]": current_interpolant})

# create and run simulation using the CasadiSolver in "fast" mode, remembering to
# pass in the updated parameters
sim = pybamm.Simulation(
    model, parameter_values=param, solver=pybamm.CasadiSolver(mode="safe", dt_max= 0.01, extra_options_setup={"max_num_steps": 1000})
)
solution = sim.solve()


solution.save_data(
    "thermal_pulse_test_sim.csv",
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
    ],
    to_format="csv",
)

plot = pybamm.QuickPlot(
    solution,
    [
        "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "Positive particle surface concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Terminal voltage [V]",
        "X-averaged cell temperature [K]",
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()

