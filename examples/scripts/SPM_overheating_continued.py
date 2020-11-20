#
# Example of SPM with decomposition reactions at high temperatures with continued TR
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

operating_mode = ExternalCircuitResistanceFunction() 

options1 = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    # "operating mode": operating_mode, 
}
options2 = {
    "thermal": "x-lumped",
    "operating mode": operating_mode
}
options3 = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    "operating mode": operating_mode, 
    "external submodels": ["negative particle", "positive particle"],
}
options4 = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    # "operating mode": operating_mode, 
    "external submodels": ["negative particle"],
}
options5 = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    # "operating mode": operating_mode, 
    "external submodels": ["negative particle", "positive particle"],
}
models = [
    pybamm.lithium_ion.SPM(options1, name="with decomposition"),
    pybamm.lithium_ion.SPM(options3, name="with decomposition thermal continued"),
    # pybamm.lithium_ion.SPM(options2, name="without decomposition"),
]

solutions = []
for model in models:   

    # set initial conditions for continued model 
    if model.name == "with decomposition thermal continued":
        # c_n = model.variables["X-averaged negative particle concentration"]
        # c_p = model.variables["X-averaged positive particle concentration"]
        I = model.variables["Total current density"]
        Q = model.variables["Discharge capacity [A.h]"]
        T = model.variables["Volume-averaged cell temperature"]
        z = model.variables["Relative SEI thickness"]
        alpha = model.variables["Degree of conversion of cathode decomposition"]
        c_an = model.variables["Fraction of Li in SEI"]
        
        model.initial_conditions = {
            # c_n: pybamm.Vector(solutions[0]["X-averaged negative particle concentration"].data[:,-1]),
            # c_p: pybamm.Vector(solutions[0]["X-averaged positive particle concentration"].data[:,-1]),
            I: pybamm.Scalar(solutions[0]["Total current density"].data[-1]),
            Q: pybamm.Scalar(solutions[0]["Discharge capacity [A.h]"].data[-1]),
            T: pybamm.Scalar(solutions[0]["Volume-averaged cell temperature"].data[-1]),
            z: pybamm.Scalar(solutions[0]["Relative SEI thickness"].data[-1]),
            alpha: pybamm.Scalar(solutions[0]["Degree of conversion of cathode decomposition"].data[-1]),
            c_an: pybamm.Scalar(solutions[0]["Fraction of Li in SEI"].data[-1]),

            }

    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
        # "Cell capacity [A.h]": 10,  #match Kriston et al.
        # "Typical current [A]": 10, #match Kriston et al.
        "Ambient temperature [K]": 390, 
        "Initial temperature [K]": 390, 
        "Resistance [ohm]": 0.01,
        "Edge heat transfer coefficient [W.m-2.K-1]": 30,
        "Separator thickness [m]":0.0015*4,
        },
        check_already_exists=False,
    )
    if model.name == "with decomposition thermal continued": 
        param.update({
            "Resistance [ohm]": 1, # to stop the current
        })
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
            
    # solve model 
    t_end = 60 # for T_amb = 390
    t_eval = np.linspace(0,t_end, 3000)
    if model.name == "with decomposition thermal continued":
        t_eval = np.linspace(t_end,t_end*10, 3000)

        # set particle concentrations as constants
        c_s_n_xav = solutions[0]["X-averaged negative particle concentration"].data[:,-1][:,np.newaxis]
        c_s_p_xav = solutions[0]["X-averaged positive particle concentration"].data[:,-1][:,np.newaxis]

        external_variables = {"X-averaged positive particle concentration":c_s_p_xav,
            "X-averaged negative particle concentration":c_s_n_xav,
            }
        # sim.step(dt, external_variables = external_variables)
        solution = pybamm.ScikitsDaeSolver().solve(model, t_eval, external_variables = external_variables)
        # solution = model.default_solver.solve(model, t_eval, external_variables = external_variables)



    else:
        solution = pybamm.ScikitsDaeSolver().solve(model, t_eval)
        # solution = model.default_solver.solve(model, t_eval)

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
        "X-averaged negative electrode extent of lithiation",        
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
