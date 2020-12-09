#
# Example of SPM with decomposition reactions at high temperatures with continued TR with constant positive particle concentration. 
#
# NOTE: For solver integration error, reduce the t_eval endtime.

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model

options1 = {
    "thermal": "two-state lumped",
}

model = pybamm.lithium_ion.SPM(options1, name="Mohtat2020")

experiment = pybamm.Experiment(
    [
        "Charge at 1 A until 4.2 V",
        "Hold at 4.2 V until 50 mA",
        "Rest for 1 hour",
    ]
)

solver = pybamm.CasadiSolver(mode="safe", extra_options_setup={"max_num_steps": 1000})
sim = pybamm.Simulation(model, experiment=experiment, solver=solver)
sol = sim.solve()

plot = pybamm.QuickPlot(
    sol,
    [   "Current [A]",
        "Terminal voltage [V]",
        "X-averaged negative particle concentration [mol.m-3]",
        "X-averaged positive particle concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        "X-averaged cell temperature [K]",     
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()

