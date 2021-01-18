#
# Example of SPM with decomposition reactions at high temperatures with continued TR with constant positive particle concentration. 
#
# NOTE: For solver integration error, reduce the t_eval endtime.

import pybamm 
import numpy as np
from shutil import copy


pybamm.set_logging_level("INFO")

# load model
class ExternalCircuitResistanceFunction():
    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return V - I*pybamm.FunctionParameter("Resistance [ohm]", {"Time [s]": pybamm.t})


def pulse_test(pulse_time, rest_time, pulse_current):
    def current(t):
        floor = pybamm.Function(np.floor, t/(pulse_time + rest_time))
        mod_t = t-(pulse_time + rest_time)*floor
        pulse_signal = mod_t < pulse_time
        return pulse_signal * pulse_current
    return current

operating_mode = ExternalCircuitResistanceFunction() 

options1 = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    "operating mode": operating_mode,
    "kinetics": "modified BV" 
}

model = pybamm.lithium_ion.SPM(options1, name="with decomposition")


# model.events={}

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
soc_0 = 1
param.update(
    {
    "Lower voltage cut-off [V]": 0,
    "Resistance [ohm]": 0.017, #0.011, 
    
    "Cell capacity [A.h]": 4.6, 
    "Typical current [A]": 4.6,
    "Ambient temperature [K]":296.7,
    "Initial temperature [K]": 296.7,
    "Initial concentration in negative electrode [mol.m-3]":(soc_0*(0.87-0.0017)+0.0017)*28746, #x0 (0.0017) * Csmax_n(28746)
    "Initial concentration in positive electrode [mol.m-3]":(0.8907-soc_0*(0.8907-0.03))*35380, #y0 (0.8907) * Csmax_p(35380)

    "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 10,  
    "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 10,  
    "Negative tab heat transfer coefficient [W.m-2.K-1]":20,  
    "Positive tab heat transfer coefficient [W.m-2.K-1]":20,  
    "Edge heat transfer coefficient [W.m-2.K-1]":20,
    "Total heat transfer coefficient [W.m-2.K-1]":20,
    "Negative electrode thickness [m]":62E-06*4.2/5, 
    "Positive electrode thickness [m]":67E-06*4.2/5,
    # "Negative electrode diffusion coefficient [m2.s-1]":5.0E-16,
    "Positive particle radius [m]": 3.5E-06*10,
    },
    check_already_exists=False,
)

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
}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
        
# solve model 
t_end = [100]
t_eval = np.linspace(0,t_end[0], 5000)
solver = pybamm.CasadiSolver(mode="safe", dt_max= 0.0001, extra_options_setup={"max_num_steps": 1000})
solution = solver.solve(model, t_eval)


#save data to csv and copy to a different folder for matlab processing 
filename = "ESC_17mOhm_100SOC.csv"
solution.save_data(
    filename,
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
    ],
    to_format="csv",
)
src = "C:/Users/Vivian/Documents/PyBaMM/" + filename 
dst = "C:/Users/Vivian/Box/Research/ESC modeling/ESC/Sim/" + filename
copy(src, dst)

plot = pybamm.QuickPlot(
    solution,
    [   "Current [A]",
        "Terminal voltage [V]",
        "X-averaged negative particle concentration",
        "X-averaged positive particle concentration",
        "Positive electrolyte concentration [mol.m-3]",
        "X-averaged electrolyte concentration [mol.m-3]",
        # "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        # "Positive particle surface concentration [mol.m-3]",
        # "Negative electrode potential [V]",
        # "Electrolyte potential [V]",
        # "Positive electrode potential [V]",
        # "Anode decomposition reaction rate",
        # "Cathode decomposition reaction rate",
        "X-averaged cell temperature [K]",
        # "Surface cell temperature [K]",
        # "Ambient temperature [K]",
        # "Relative SEI thickness",
        "Fraction of Li in SEI",
        # "Degree of conversion of cathode decomposition",
        # "Anode decomposition heating [W.m-3]",
        # "Cathode decomposition heating [W.m-3]",
        "SEI decomposition heating [W.m-3]",
        # "X-averaged Ohmic heating [W.m-3]",
        "X-averaged irreversible electrochemical heating [W.m-3]",
        # "X-averaged total heating [W.m-3]",
        "X-averaged negative electrode extent of lithiation",     
        # "Exchange current density [A.m-2]",           
        # "Core-surface temperature difference [K]"
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()



