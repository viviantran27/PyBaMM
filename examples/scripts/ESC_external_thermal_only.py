
# run thermal model only

import pybamm 
import numpy as np
from shutil import copy
import pandas
from scipy.interpolate import interp1d

pybamm.set_logging_level("INFO")

# 1. initialise the moel
model = pybamm.lithium_ion.SPMe() 
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
    "external submodels": []
}
# 2. define parameters and variables
filename = "ESC_SPMe_thermal_input_h1-5_Cp1-5_R016.csv"
model = pybamm.lithium_ion.SPMe(options)
soc_0 = 1
R_total = 0.016 #0.016 original, min R for 600s is 0.0243
h = 1.5
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

param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
param.update(
    {
    "Cell capacity [A.h]": 4.6, #nominal
    "Typical current [A]": 4.6,
    "Ambient temperature [K]":296.7,
    "Initial temperature [K]": 296.7,
    "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": h,  
    "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": h,  
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100*Cp,
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 1100*Cp,
    },
    check_already_exists=False,
)
# param["Current function [A]"] = "[current data]ESC_100SOC_test" # uncomment to use measured ESC current 

geometry = model.default_geometry
T_vol_av = pybamm.standard_variables.T_vol_av
T_amb = 273.15 #param.T_amb(pybamm.t * param.timescale)
Q_vol_av = 5 #Q_ohm + Q_rxn + Q_rev

cell_volume = param.l * param.l_y * param.l_z

yz_cell_surface_area = param.l_y * param.l_z
yz_surface_cooling_coefficient = (
    -(param.h_cn + param.h_cp)
    * yz_cell_surface_area
    / cell_volume
    / (param.delta ** 2)
)

negative_tab_area = param.l_tab_n * param.l_cn
negative_tab_cooling_coefficient = (
    -param.h_tab_n * negative_tab_area / cell_volume / param.delta
)

positive_tab_area = param.l_tab_p * param.l_cp
positive_tab_cooling_coefficient = (
    -param.h_tab_p * positive_tab_area / cell_volume / param.delta
)

edge_area = (
    2 * param.l_y * param.l
    + 2 * param.l_z * param.l
    - negative_tab_area
    - positive_tab_area
)
edge_cooling_coefficient = (
    -param.h_edge * edge_area / cell_volume / param.delta
)

h_total = (
    yz_surface_cooling_coefficient
    + negative_tab_cooling_coefficient
    + positive_tab_cooling_coefficient
    + edge_cooling_coefficient
)


# 3. state governing equations
model.rhs = {
    T_vol_av: (
        param.B * Q_vol_av + h_total * (T_vol_av - T_amb)
    )
    / (param.C_th * param.rho(T_vol_av))
}

# 4. state boundary conditions
# 5. state initial conditions
model.initial_conditions = {T_vol_av: param.T_init}

# 6. state output variables
Q_scale = param.i_typ * param.potential_scale / param.L_x
model.variables = {"Volume-averaged temperature": T_vol_av, "Volume-averaged temperature [K]": T_vol_av * Q_scale}

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
t_eval = np.linspace(0,t_end[0], 1000)
solver = pybamm.CasadiSolver(mode="safe", dt_max= 0.001, extra_options_setup={"max_num_steps": 10000})
solution = solver.solve(model, t_eval)
