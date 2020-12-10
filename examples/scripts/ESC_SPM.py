#
# Example of SPM with decomposition reactions at high temperatures with continued TR with constant positive particle concentration. 
#
# NOTE: For solver integration error, reduce the t_eval endtime.

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
class ExternalCircuitResistanceFunction():
    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return V / I - pybamm.FunctionParameter("Resistance [ohm]", {"Time [s]": pybamm.t})


def pulse_test(pulse_time, rest_time, pulse_current):
    def current(t):
        floor = pybamm.Function(np.floor, t/(pulse_time + rest_time))
        mod_t = t-(pulse_time + rest_time)*floor
        pulse_signal = mod_t < pulse_time
        return pulse_signal * pulse_current
    return current

operating_mode = ExternalCircuitResistanceFunction() 

options1 = {
    "thermal": "two-state lumped",
    # "side reactions": "decomposition",
    "operating mode": operating_mode,
    "kinetics": "modified BV" 
}

model = pybamm.lithium_ion.SPM(options1, name="with decomposition")


# model.events={}

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
param.update(
    {
    "Cell capacity [A.h]": 10, #match Kriston et al.
    "Typical current [A]": 10, #match Kriston et al.
    "Lower voltage cut-off [V]": 0,

    "Resistance [ohm]": 0.03, #0.011, #Rint=~1.5mOhm
    # "Edge heat transfer coefficient [W.m-2.K-1]":10,
    # "Negative tab heat transfer coefficient [W.m-2.K-1]":10,
    # "Positive tab heat transfer coefficient [W.m-2.K-1]":10,
    # "Total tab heat transfer coefficient [W.m-2.K-1]":10,
    # "Cell cooling surface area [m2]": 0.41,

    # "Frequency factor for SEI decomposition [s-1]":2.25E15, #2.25E15 default
    # "Frequency factor for cathode decomposition [s-1]":2.55E14, #2.55E14 default
    # "Frequency factor for anode decomposition [s-1]":2.5E13, #2.5E13 default
    "Activation energy for SEI decomposition [J]":2.03E-19,
    "Activation energy for anode decomposition [J]":2.03E-19,
    "Activation energy for cathode decomposition [J]": 2.1E-19,
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
t_end = [3600*1]
t_eval = np.linspace(0,t_end[0], 5000)
solution = model.default_solver.solve(model, t_eval)



# save data
# solutions[0].save_data(
#     "ESC_SPM.csv",
#     [
#         "Time [h]",
#         "Current [A]",
#         "Terminal voltage [V]",
#         "Discharge capacity [A.h]",
#         "X-averaged cell temperature [K]",
#     ],
#     to_format="csv",
# )
# solutions[1].save_data(
#     "ESC_SPM1.csv",
#     [
#         "Time [h]",
#         "Current [A]",
#         "Terminal voltage [V]",
#         "Discharge capacity [A.h]",
#         "X-averaged cell temperature [K]",
#     ],
#     to_format="csv",
# )

# print("Done saving data to csv.")

# plot
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
        "Surface cell temperature [K]",
        # "Ambient temperature [K]",
        # "Relative SEI thickness",
        # "Fraction of Li in SEI",
        # "Degree of conversion of cathode decomposition",
        "Anode decomposition heating [W.m-3]",
        "Cathode decomposition heating [W.m-3]",
        "SEI decomposition heating [W.m-3]",
        # "X-averaged Ohmic heating [W.m-3]",
        "X-averaged irreversible electrochemical heating [W.m-3]",
        # "X-averaged total heating [W.m-3]",
        "X-averaged negative electrode extent of lithiation",     
        # "Exchange current density [A.m-2]",           
        "Core-surface temperature difference [K]"
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()

