#
# Simulate thermal pulse test (inputed as a drive cycle) loaded from csv file
#
import pybamm
import pandas as pd
import os
import numpy as np

os.chdir(pybamm.__path__[0] + "/..")

pybamm.set_logging_level("INFO")

# load model and update parameters so the input current is the drive cycle
model = pybamm.lithium_ion.SPM({"cell geometry": "pouch", "thermal": "x-lumped"}) 
# model = pybamm.lithium_ion.SPM({"cell geometry": "arbitrary", "thermal": "lumped"}) 
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
soc_0 = 0.5
param.update(
    {    
    "Cell capacity [A.h]": 4.6, 
    "Typical current [A]": 4.6,
    "Ambient temperature [K]":296.7,
    "Initial temperature [K]": 296.7,
    "Initial concentration in negative electrode [mol.m-3]":(soc_0*(0.87-0.0017)+0.0017)*28746, #x0 (0.0017) * Csmax_n(28746)
    "Initial concentration in positive electrode [mol.m-3]":(0.8907-soc_0*(0.8907-0.03))*35380, #y0 (0.8907) * Csmax_p(35380)

    "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 30,  
    "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 30,  
    "Negative tab heat transfer coefficient [W.m-2.K-1]":30,  
    "Positive tab heat transfer coefficient [W.m-2.K-1]":30,  
    "Edge heat transfer coefficient [W.m-2.K-1]":30,
    "Total heat transfer coefficient [W.m-2.K-1]":30,
    "Negative electrode thickness [m]":62E-06*4.2/5, 
    "Positive electrode thickness [m]":67E-06*4.2/5,
    }
)

# import drive cycle from file
drive_cycle = pd.read_csv(
    "pybamm/input/drive_cycles/thermal_pulse_test.csv", comment="#", header=None
).to_numpy()

# create interpolant
timescale = param.evaluate(model.timescale)
current_interpolant = pybamm.Interpolant(drive_cycle, timescale * pybamm.t)
param["Current function [A]"] = current_interpolant

# create and run simulation using the CasadiSolver in "fast" mode, remembering to
# pass in the updated parameters
sim = pybamm.Simulation(
    model, parameter_values=param, solver=pybamm.CasadiSolver(mode="safe", dt_max= 0.1, extra_options_setup={"max_num_steps": 1000})
)
# t_end = [2250]
# t_eval = np.linspace(0,t_end[0], 5000)
# solution = sim.solve(t_eval)
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

