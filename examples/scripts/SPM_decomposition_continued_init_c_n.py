#
# Example of SPM with decomposition reactions at high temperatures with continued TR with c_n initialized. 
#
# NOTE: For solver integration error, reduce the t_eval endtime
# .
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
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    "operating mode": operating_mode, 
}
options2 = {
    "thermal": "x-lumped",
    "operating mode": operating_mode
}
options3 = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    "operating mode": operating_mode, 
    "external submodels": ["positive particle"],
}
options4 = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    "operating mode": operating_mode, 
    "external submodels": ["negative particle", "positive particle"],
}
models = [
    pybamm.lithium_ion.SPM(options1, name="with decomposition"),
    pybamm.lithium_ion.SPM(options3, name="with decomposition thermal continued"),
    # pybamm.lithium_ion.SPM(options3, name="with decomposition thermal continued constant particle concentrations"),
    # pybamm.lithium_ion.SPM(options2, name="without decomposition"),
]

solutions = []
for model in models:   
    # model.events={}

    # set initial conditions for continued model 
    if model.name == "with decomposition thermal continued":
        c_n = model.variables["X-averaged negative particle concentration"]
        # c_p = model.variables["X-averaged positive particle concentration"]
        I = model.variables["Total current density"]
        Q = model.variables["Discharge capacity [A.h]"]
        T = model.variables["Volume-averaged cell temperature"]
        z = model.variables["Relative SEI thickness"]
        alpha = model.variables["Degree of conversion of cathode decomposition"]
        c_an = model.variables["Fraction of Li in SEI"]
        
        model.initial_conditions = {
            c_n: pybamm.Vector(solutions[0]["X-averaged negative particle concentration"].data[:,-1]),
            # c_p: pybamm.Vector(solutions[0]["X-averaged positive particle concentration"].data[:,-1]),
            I: pybamm.Scalar(solutions[0]["Total current density"].data[-1]),
            Q: pybamm.Scalar(solutions[0]["Discharge capacity [A.h]"].data[-1]),
            T: pybamm.Scalar(solutions[0]["Volume-averaged cell temperature"].data[-1]),
            z: pybamm.Scalar(solutions[0]["Relative SEI thickness"].data[-1]),
            alpha: pybamm.Scalar(solutions[0]["Degree of conversion of cathode decomposition"].data[-1]),
            c_an: pybamm.Scalar(solutions[0]["Fraction of Li in SEI"].data[-1]),
            }

    # set initial conditions for second continued model 
    if model.name == "with decomposition thermal continued constant particle concentrations":
        c_n = model.variables["X-averaged negative particle concentration"]
        # c_p = model.variables["X-averaged positive particle concentration"]
        I = model.variables["Total current density"]
        Q = model.variables["Discharge capacity [A.h]"]
        T = model.variables["Volume-averaged cell temperature"]
        z = model.variables["Relative SEI thickness"]
        alpha = model.variables["Degree of conversion of cathode decomposition"]
        c_an = model.variables["Fraction of Li in SEI"]
        
        model.initial_conditions = {
            c_n: pybamm.Vector(solutions[1]["X-averaged negative particle concentration"].data[:,-1]),
            # c_p: pybamm.Vector(solutions[1]["X-averaged positive particle concentration"].data[:,-1]),
            I: pybamm.Scalar(solutions[1]["Total current density"].data[-1]),
            Q: pybamm.Scalar(solutions[1]["Discharge capacity [A.h]"].data[-1]),
            T: pybamm.Scalar(solutions[1]["Volume-averaged cell temperature"].data[-1]),
            z: pybamm.Scalar(solutions[1]["Relative SEI thickness"].data[-1]),
            alpha: pybamm.Scalar(solutions[1]["Degree of conversion of cathode decomposition"].data[-1]),
            c_an: pybamm.Scalar(solutions[1]["Fraction of Li in SEI"].data[-1]),
            }
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
        # "Maximum concentration in positive electrode [mol.m-3]":63104*1.05,
        # "Cell capacity [A.h]": 10, #match Kriston et al.
        # "Typical current [A]": 10, #match Kriston et al.
        "Lower voltage cut-off [V]": 0,
        "Ambient temperature [K]": 390, 
        "Initial temperature [K]": 390, 
        "Resistance [ohm]": 1, #0.011, #Rint=~1.5mOhm
        # "Edge heat transfer coefficient [W.m-2.K-1]": 1000,

        # "Frequency factor for SEI decomposition [s-1]":2.25E15, #2.25E15 default
        # "Activation energy for SEI decomposition [J]":2.24E-16, #2.24E-19 default
        # "Enthalpy of SEI decomposition [J.kg-1]":257000,
        },
        check_already_exists=False,
    )
    if model.name == "with decomposition thermal continued constant particle concentrations": 
        param.update({
            # "Resistance [ohm]": 0.1, # to stop the current
            # "Negative electrode diffusivity [m2.s-1]": 3.3E-13, 
            # "Separator porosity":0.47,
            # "Electrolyte conductivity [S.m-1]":1,
            # "Electrolyte diffusivity [m2.s-1]": 1E-3,
        })
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
        var.y: 10*scale,
        var.z: 10*scale,
    }
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
            
    # solve model 
    # t_end = [14, 35, 60] #R = 0.014, both T = default
    # t_end = [11.2, 60*2] #R = 0.010, both T = default, only first 2 sims
    # t_end = [11.2, 23.5, 60] #R = 0.010, both T = default
    # t_end = [11.2, 23.5, 60*20] #R = 0.010, both T = default, without decomp
    # t_end = [13, 37, 100] #R = 0.014, both T = 390K
    # t_end = [2000] #R = 1, both T = 390K
    # t_end = [11.2, 300] #R = 0.010, both T = default, edge heat transfer coeff = 1500
    t_end = [60*14.5, 60*15]

    t_eval = np.linspace(0,t_end[0], 5000)

    if model.name == "with decomposition thermal continued":
        t_eval = np.linspace(t_end[0],t_end[1], 5000)

        # set particle concentrations as constants
        # c_s_n_xav = solutions[0]["X-averaged negative particle concentration"].data[:,-1][:,np.newaxis]
        c_s_p_xav = solutions[0]["X-averaged positive particle concentration"].data[:,-1][:,np.newaxis]

        external_variables = {"X-averaged positive particle concentration":c_s_p_xav,
            # "X-averaged negative particle concentration":c_s_n_xav,
            }
        # sim.step(dt, external_variables = external_variables)
        solution = pybamm.ScikitsDaeSolver().solve(model, t_eval, external_variables = external_variables)
        # solution = model.default_solver.solve(model, t_eval, external_variables = external_variables)

    elif model.name == "with decomposition thermal continued constant particle concentrations":
        t_eval = np.linspace(t_end[1],t_end[2], 3000)

        # set particle concentrations as constants
        # c_s_n_xav = solutions[1]["X-averaged negative particle concentration"].data[:,-1][:,np.newaxis]
        c_s_p_xav = solutions[1]["X-averaged positive particle concentration"].data[:,-1][:,np.newaxis]

        external_variables = {"X-averaged positive particle concentration":c_s_p_xav,
            # "X-averaged negative particle concentration":c_s_n_xav,
            }
        # sim.step(dt, external_variables = external_variables)
        solution = pybamm.ScikitsDaeSolver().solve(model, t_eval, external_variables = external_variables)
        # solution = model.default_solver.solve(model, t_eval, external_variables = external_variables)


    else:
        solution = pybamm.ScikitsDaeSolver().solve(model, t_eval)
        # solution = model.default_solver.solve(model, t_eval)

    solutions.append(solution)

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
# # solutions[2].save_data(
# #     "ESC_SPM2.csv",
# #     [
# #         "Time [h]",
# #         "Current [A]",
# #         "Terminal voltage [V]",
# #         "Discharge capacity [A.h]",
# #         "X-averaged cell temperature [K]",
# #     ],
# #     to_format="csv",
# # )
# print("Done saving data to csv.")
# plot
plot = pybamm.QuickPlot(
    solutions,
    [   "Current [A]",
        "Terminal voltage [V]",
        # "X-averaged negative particle concentration",
        # "X-averaged positive particle concentration",
        # "Negative particle surface concentration [mol.m-3]",
        # "Electrolyte concentration [mol.m-3]",
        # "Positive particle surface concentration [mol.m-3]",
        # "Negative electrode potential [V]",
        # "Electrolyte potential [V]",
        # "Positive electrode potential [V]",
        # "Anode decomposition reaction rate",
        # "Cathode decomposition reaction rate",
        "X-averaged cell temperature [K]",
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
        "Negative electrode average extent of lithiation",     
        # "Exchange current density [A.m-2]",   
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()

