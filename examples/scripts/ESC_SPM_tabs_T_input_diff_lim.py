#
# SPM model with ESC + tabbing resistance, run electrochemical model with temperature input. 
# Only runs ~2 secs with ESC. Runs longer without. Doesn't seem like any other heating is active...
# NOTE: For solver integration error, reduce the t_eval endtime.

import pybamm 
import numpy as np
from shutil import copy
import pandas
from scipy.interpolate import interp1d

pybamm.set_logging_level("INFO")

# calculate load profile for constant resistance (R_ext + R_tab)
model = pybamm.lithium_ion.SPMe() # pre-define model to get timescale in function
class ExternalCircuitResistanceFunction():
    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        R_ext = pybamm.FunctionParameter("External resistance [Ohm]", {"Time [s]": pybamm.t * model.param.timescale})
        R_tab = pybamm.FunctionParameter("Tabbing resistance [Ohm]", {"Time [s]": pybamm.t * model.param.timescale})
        return V/I - (R_ext + R_tab)

# choose submodels 
options = {
    "thermal": "x-lumped",
    # "side reactions": "decomposition",
    "operating mode": ExternalCircuitResistanceFunction(),
    # "surface form": "differential"
    "external submodels": ["thermal"]
}

filename = "ESC_SPMe_thermal_input_h1-5_Cp2-5_R016.csv"
model = pybamm.lithium_ion.SPMe(options)
soc_0 = 1
R_total = 0.016 #0.016 original, min R for 600s is 0.0243
h = 1.5
Cp = 2.5

# add variable to confirm actual resistance is constant
V = model.variables["Terminal voltage [V]"]
I = model.variables["Current [A]"]
R_tab = pybamm.Parameter("Tabbing resistance [Ohm]")
R_ext = pybamm.Parameter("External resistance [Ohm]")


model.variables.update({
    "Terminal voltage [V]": V - I*R_tab,
    "Actual resistance [Ohm]":V/I,
    }
)

# load parameter values and process model and geometry
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)

# define temperature profile
temperature_profile = pandas.read_csv("pybamm/input/drive_cycles/ESC_100SOC_temperature.csv", comment="#", header=None).to_numpy()

param.update(
    {
    # "Temperature function [K]": temperature_interpolant,
    "Tabbing resistance [Ohm]": 0.009, 
    "External resistance [Ohm]": R_total -0.009, # 0.016 matches 100% SOC ESC data

    "Lower voltage cut-off [V]": 0,     
    "Nominal cell capacity [A.h]": 4.6, 
    "Typical current [A]": 4.6,
    "Ambient temperature [K]":297.2909077,
    "Initial temperature [K]": 297.2909077,
    "Initial concentration in negative electrode [mol.m-3]":(soc_0*(0.87-0.0017)+0.0017)*28746, #x0 (0.0017) * Csmax_n(28746)
    "Initial concentration in positive electrode [mol.m-3]":(0.8907-soc_0*(0.8907-0.03))*35380, #y0 (0.8907) * Csmax_p(35380)

    "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": h,  
    "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": h,  
    "Negative electrode thickness [m]":62E-06 * 4.2/5, 
    "Positive electrode thickness [m]":67E-06 * 4.2/5,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100*Cp,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 1100*Cp,

    },
    check_already_exists=False,
)

# solve model 
sim = pybamm.Simulation(model, parameter_values=param)
sim.solver = pybamm.CasadiSolver(mode="safe", dt_max= 0.001, extra_options_setup={"max_num_steps": 10000})
t_eval = np.linspace(0,600, 10000)

# define temperature change with time
T_ref = param.evaluate(model.param.T_ref)
Delta_T = param.evaluate(model.param.Delta_T)
T_av_dim = interp1d(temperature_profile[:,0], temperature_profile[:,1], kind = 'linear')
for i in np.arange(1, len(t_eval) - 1): 
    dt = t_eval[i + 1] - t_eval[i]
    T_av = (T_av_dim(t_eval[i]) - T_ref) / Delta_T
    external_variables = {
        "Volume-averaged cell temperature": T_av, 
        }
    solution = sim.step(dt, external_variables=external_variables)

# save data to csv and copy to a different folder for matlab processing 
# 
sim.solution.save_data(
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
dst = "C:/Users/Vivian/Box/Research/ESC modeling/ESC/Sim/" + filename
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
