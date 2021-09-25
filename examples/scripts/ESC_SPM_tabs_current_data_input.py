#
# SPM model with ESC + tabbing resistance
# Current data input, but doesn't run b/c can't find states

import pybamm 
import numpy as np
import pandas as pd
from shutil import copy


pybamm.set_logging_level("INFO")

# calculate load profile for constant resistance (R_ext + R_tab)
param=pybamm.LithiumIonParameters() # pre-define param to get timescale in function 
class ExternalCircuitResistanceFunction():
    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        R_ext = pybamm.FunctionParameter("External resistance [Ohm]", {"Time [s]": pybamm.t * param.timescale})
        R_tab = pybamm.FunctionParameter("Tabbing resistance [Ohm]", {"Time [s]": pybamm.t * param.timescale})
        return V/I - (R_ext + R_tab)

# choose submodels 
options = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    # "operating mode": ExternalCircuitResistanceFunction(),
    # "surface form": "differential"
}


# filename = "ESC_SPMe_9mOhm_100SOC_h1-5_R16mOhm_1-5Cp_Dp_arr_0-4.csv"
model = pybamm.lithium_ion.SPMe(options)
soc_0 = 1
h = 1.5
R_total = 0.016
Cp = 1.5

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

# model.events={} #ignore all events

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
param.update(
    {
    "Tabbing resistance [Ohm]": 0.009, 
    "External resistance [Ohm]": R_total -0.009, # 0.016 matches 100% SOC ESC data

    "Lower voltage cut-off [V]": 0,     
    "Cell capacity [A.h]": 4.6, #nominal
    "Typical current [A]": 4.6,
    "Ambient temperature [K]":296.7,
    "Initial temperature [K]": 296.7,
    "Initial concentration in negative electrode [mol.m-3]":(soc_0*(0.87-0.0017)+0.0017)*28746, #x0 (0.0017) * Csmax_n(28746)
    "Initial concentration in positive electrode [mol.m-3]":(0.8907-soc_0*(0.8907-0.03))*35380, #y0 (0.8907) * Csmax_p(35380)

    "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": h,  
    "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": h,  
    # "Negative tab heat transfer coefficient [W.m-2.K-1]":20,  
    # "Positive tab heat transfer coefficient [W.m-2.K-1]":20,  
    # "Edge heat transfer coefficient [W.m-2.K-1]":20,
    # "Total heat transfer coefficient [W.m-2.K-1]":20,
    "Negative electrode thickness [m]":62E-06 * 4.2/5, 
    "Positive electrode thickness [m]":67E-06 * 4.2/5,
    # "Negative electrode diffusion coefficient [m2.s-1]":5.0E-15,
    # "Positive particle radius [m]": 3.5E-06*0.3,
    # "Positive electrode diffusivity [m2.s-1]":"[function]NMC_diffusivity_PeymanMPM",

    # "Negative current collector conductivity [S.m-1]": 59600000*0.005,
    # "Positive current collector conductivity [S.m-1]": 35500000*0.005,

    "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100*Cp,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 1100*Cp,

    "Frequency factor for anode decomposition [s-1]":2.5E13,
    "Activation energy for anode decomposition [J]":2.24E-19,
    "Enthalpy of anode decomposition [J.kg-1]":1714000,

    },
    check_already_exists=False,
)
# param["Current function [A]"] = "[current data]ESC_100SOC_test" # uncomment to use measured ESC current 
drive_cycle = pd.read_csv("pybamm/input/drive_cycles/ESC_100SOC_current.csv", comment="#", header=None).to_numpy()
timescale = param.evaluate(model.timescale)
current_interpolant = pybamm.Interpolant(drive_cycle[:, 0], drive_cycle[:, 1], timescale * pybamm.t)
param["Current function [A]"] = current_interpolant

param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
scale = 2
var_pts =  {
    var.x_n: 20*scale,
    var.x_s: 20*scale,
    var.x_p: 20*scale,
    var.r_n: 10*scale,
    var.r_p: 10*scale,
    var.z: 20*scale,
}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
        
# solve model 
t_end = [600]
t_eval = np.linspace(0,t_end[0], 10000)
solver = pybamm.CasadiSolver(mode="safe", dt_max= 0.001, extra_options_setup={"max_num_steps": 10000})
solution = solver.solve(model, t_eval)


# save data to csv and copy to a different folder for matlab processing 
# solution.save_data(
#     filename,
#     [
#         "Time [h]",
#         "Current [A]",
#         "Terminal voltage [V]",
#         "Discharge capacity [A.h]",
#         "Volume-averaged cell temperature [K]",
#     ],
#     to_format="csv",
# )

# src = "C:/Users/Vivian/Documents/PyBaMM/" + filename 
# dst = "C:/Users/Vivian/Box/Research/ESC modeling/ESC/Sim/" + filename
# copy(src, dst)

# plot simulation results
plot = pybamm.QuickPlot(
    solution,
    [   "Current [A]",
        "Terminal voltage [V]",
        # "Tab voltage [V]",
        # "X-averaged negative particle concentration",
        "X-averaged positive particle concentration",
        "X-averaged negative particle concentration",
        # # "Positive electrolyte concentration [mol.m-3]",
        # "X-averaged electrolyte concentration [mol.m-3]",
        # # "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        # "Positive particle surface concentration [mol.m-3]",
        # # "Negative electrode potential [V]",
        # # "Electrolyte potential [V]",
        # # "Positive electrode potential [V]",
        # "Anode decomposition reaction rate",
        # "Cathode decomposition reaction rate",
        # "X-averaged cell temperature [K]",
        # # "Surface cell temperature [K]",
        # # "Ambient temperature [K]",
        # # "Relative SEI thickness",
        # "Fraction of Li in SEI",
        # # "Degree of conversion of cathode decomposition",
        # "Anode decomposition heating [W.m-3]",
        # "Cathode decomposition heating [W.m-3]",
        "SEI decomposition heating [W.m-3]",
        ["Volume-averaged Ohmic heating [W.m-3]",
        "Volume-averaged irreversible electrochemical heating [W.m-3]",
        "Volume-averaged total heating [W.m-3]",],
        "X-averaged negative electrode extent of lithiation",     
        # # "Exchange current density [A.m-2]",           
        # # "Core-surface temperature difference [K]"
        "Volume-averaged cell temperature [K]",
        "Tab heating [W.m-3]",
        "Actual resistance [Ohm]",
        "Positive electrode exchange current density [A.m-2]",
        "Negative electrode exchange current density [A.m-2]",
        "Electrolyte flux", 
        "Negative electrode entropic change",
        "Positive electrode entropic change",
        "Positive electrode SOC",
        "Negative electrode SOC",

    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
