#
# Perform a pulse test with the SPM Li-ion cell. Format data and heading 
# to acceptable csv for Matlab to extract RC parameters.
#
# NOTE: For solver integration error, reduce the t_eval endtime
# .
import pybamm
import numpy as np
from shutil import copy


pybamm.set_logging_level("INFO")

# load model
def pulse_test(pulse_time, rest_time, pulse_current):
    def current(t):
        floor = pybamm.Function(np.floor, t/(pulse_time + rest_time))
        mod_t = t-(pulse_time + rest_time)*floor
        pulse_signal = mod_t < pulse_time
        return pulse_signal * pulse_current
    return current

operating_mode = "current"

options = {
    "current collector": "potential pair",
    "thermal": "x-lumped",
    # "side reactions": "decomposition",
    "operating mode": operating_mode, 
    "dimensionality":1,
}
models = [
    pybamm.lithium_ion.SPM(options, name="SPM"),
]

solutions = []
for model in models:
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    soc_0 = 1
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
        # "Cell capacity [A.h]": 0.5, 
        # "Typical current [A]": 0.5,
        "Current function [A]": pulse_test(2*60, 5*60, 9),
        "Edge heat transfer coefficient [W.m-2.K-1]": 3000,
        "Negative electrode thickness [m]":62E-06*100, # cell 43 
        "Positive electrode thickness [m]":67E-06*100,
        # "Separator thickness [m]":12E-06,
        # "Positive electrode conductivity [S.m-1]":100,
        # "Negative electrode conductivity [S.m-1]":100,
        # "Positive particle radius [m]": 3.5E-06*2,
        # "Negative particle radius [m]":2.5E-06*2,
        "Initial concentration in negative electrode [mol.m-3]": (soc_0*(0.87-0.0017)+0.0017)*28746, #x0 (soc_0*(0.87-0.0017)+0.0017)*28746 (0.0017) * Csmax_n(28746)
        "Initial concentration in positive electrode [mol.m-3]": (0.8907-soc_0*(0.8907-0.03))*35380, #y0 (0.8907-soc_0*(0.8907-0.03))*35380 (0.8907) * Csmax_p(35380) 
        # "Negative electrode diffusion coefficient [m2.s-1]":5.0E-15,
        # "Positive particle radius [m]": 3.5E-06,
        "Ambient temperature [K]": 23+273.15,
        "Initial temperature [K]": 23+273.15,

        "Negative tab centre z-coordinate [m]": 0,
        "Positive tab centre z-coordinate [m]": pybamm.geometric_parameters.L_z,
        # "Negative current collector conductivity [S.m-1]": 59600000*0.1,
        # "Positive current collector conductivity [S.m-1]": 35500000*0.1,
        },
        check_already_exists=False,
    )

    var = pybamm.standard_spatial_vars
    scale = 2
    var_pts =  {
        var.x_n: 20*scale,
        var.x_s: 20*scale,
        var.x_p: 20*scale,
        var.r_n: 10*scale,
        var.r_p: 10*scale,
        var.z: 10
    }

    sim = pybamm.Simulation(model, parameter_values=param, var_pts=var_pts) 

    # solve model 
    # t_eval = np.linspace(0,5532, 3600)
    t_eval = np.linspace(0,5532, 5532*5)
    solution = sim.solve(solver=pybamm.CasadiSolver(mode="safe", dt_max= 0.001, extra_options_setup={"max_num_steps": 1000}), t_eval=t_eval)
    solutions.append(solution)

# save data to csv
solution.save_data(
    "pulse.csv",
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "Volume-averaged cell temperature [K]",
    ],
    to_format="csv",
)
src = "C:/Users/Vivian/Documents/PyBaMM/pulse.csv" 
dst = "C:/Users/Vivian/Box/Research/ESC modeling/dcr/pulse.csv"
copy(src, dst)

# plot
plot = pybamm.QuickPlot(
    solutions,
    [
        # "X-averaged negative particle concentration",
        # "X-averaged positive particle concentration",
        # "Electrolyte concentration [mol.m-3]",
        "Current [A]",
        # "Negative electrode potential [V]",
        # "Electrolyte potential [V]",
        # "Positive electrode potential [V]",
        "Terminal voltage [V]",
        "Volume-averaged cell temperature [K]",
        "X-averaged negative electrode extent of lithiation",
        
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()

