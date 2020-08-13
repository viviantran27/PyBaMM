#
# Example of SPM with decomposition reactions at high temperatures
#
# NOTE: For solver integration error, reduce the t_eval endtime
# .
import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
class ExternalCircuitResistanceFunction:

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

# operating_mode = ExternalCircuitResistanceFunction() 
operating_mode = ExternalCircuitResistanceFunction() 

options = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    "operating mode": operating_mode, 
}
models = [
    pybamm.lithium_ion.SPM(options, name="with decomposition"),
    pybamm.lithium_ion.SPM({"thermal": "x-lumped","operating mode": operating_mode}, name="without decomposition"),
]

solutions = []
for model in models:
    # turn off all events
    # model.events = []
    
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
        "Ambient temperature [K]": 390, 
        "Initial temperature [K]": 390, 
        "Resistance [ohm]": 10,
        "Edge heat transfer coefficient [W.m-2.K-1]": 30,
        # "Current function [A]": pulse_test(1*60, 5*60, 10),
        # "Positive electrode diffusivity [m2.s-1]": 3,
        # "Negative electrode diffusivity [m2.s-1]": 3,
        # "Heat transfer coefficient [W.m-2.K-1]": 0.1,
        # "Positive particle radius [m]":15E-6,
        # "Frequency factor for cathode decomposition [s-1]": 7E15,
        # "Lower voltage cut-off [V]": 2.4,
        },
        check_already_exists=False,
    )
    # param["Current function [A]"] = "[input]"
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    # solve model 
    t_eval = np.linspace(0,460, 3000)
    solution = pybamm.ScikitsDaeSolver().solve(model, t_eval)
    solutions.append(solution)

# plot
plot = pybamm.QuickPlot(
    solutions,
    [
        "X-averaged negative particle concentration",
        "X-averaged positive particle concentration",
        # "Negative particle surface concentration [mol.m-3]",
        "Electrolyte concentration [mol.m-3]",
        # "Positive particle surface concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Terminal voltage [V]",
        "Anode decomposition reaction rate",
        "Cathode decomposition reaction rate",
        "X-averaged cell temperature [K]",
        # "Ambient temperature [K]",
        "Relative SEI thickness",
        "Fraction of Li in SEI",
        "Degree of conversion of cathode decomposition",
        "Anode decomposition heating [W.m-3]",
        "Cathode decomposition heating [W.m-3]",
        "SEI decomposition heating [W.m-3]",
        "X-averaged Ohmic heating [W.m-3]",
        "X-averaged irreversible electrochemical heating [W.m-3]",
        "X-averaged total heating [W.m-3]",
        "Negative electrode average extent of lithiation",
        
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
# solution.save_data(
#     "output.csv",
#     [
#         "Time [h]",
#         "Current [A]",
#         "Terminal voltage [V]",
#         "Discharge capacity [A.h]",
#         "X-averaged cell temperature [K]",
#     ],
#     to_format="csv",
# )
