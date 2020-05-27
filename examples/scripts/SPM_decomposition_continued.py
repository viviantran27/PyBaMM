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
models = [
    pybamm.lithium_ion.SPM(options1, name="with decomposition"),
    pybamm.lithium_ion.SPM(options2, name="without decomposition"),
    pybamm.lithium_ion.SPM(options3, name="with decomposition thermal continued"),
]

solutions = []
for model in models:   
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
        "Ambient temperature [K]": 390, 
        "Initial temperature [K]": 390, 
        "Resistance [ohm]": 1,
        "Heat transfer coefficient [W.m-2.K-1]": 0.1,
        "Frequency factor for cathode decomposition [s-1]": 7E15,
        },
        check_already_exists=False,
    )
    if model.name == "with decomposition thermal continued": 
        param.update({
            # "Current function [A]": 0, 
            "Resistance [ohm]": 1, # to set current to 0
        })
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)

    if model.name == "with decomposition thermal continued": 
        model.concatenated_initial_conditions = solutions[0].y[:,-1] 
        print(solutions[0].y[:,-1])

    disc.process_model(model)

    # solve model 
    t_eval = np.linspace(0,2000, 3000)
    if model.name == "with decomposition thermal continued":
        # var = pybamm.standard_spatial_vars
        t_eval = np.linspace(1580,2000, 3000)
        # sim = pybamm.Simulation(model)
        # dt = t_eval[i+1]-t_eval[i]

        # provide particle concentrations
        # c_s_n_xav = (np.ones((10, 1))*0.5)
        c_s_p_xav = (np.ones((10, 1))*0.5)
        external_variables = {"X-averaged positive particle concentration":c_s_p_xav}
        # sim.step(dt, external_variables = external_variables)
        solution = pybamm.ScikitsDaeSolver().solve(model, t_eval, external_variables = external_variables)


    else:
        # solution = pybamm.ScikitsDaeSolver().solve(model, t_eval)
        solution = model.default_solver.solve(model, t_eval)

    solutions.append(solution)

# plot
plot = pybamm.QuickPlot(
    solutions,
    [
        # "X-averaged negative particle concentration",
        # "X-averaged positive particle concentration",
        # # "Negative particle surface concentration [mol.m-3]",
        # "Electrolyte concentration [mol.m-3]",
        # # "Positive particle surface concentration [mol.m-3]",
        # "Current [A]",
        # "Negative electrode potential [V]",
        # "Electrolyte potential [V]",
        # "Positive electrode potential [V]",
        # "Terminal voltage [V]",
        # "Anode decomposition reaction rate",
        # "Cathode decomposition reaction rate",
        "X-averaged cell temperature [K]",
        # # "Ambient temperature [K]",
        # "Relative SEI thickness",
        # "Fraction of Li in SEI",
        # "Degree of conversion of cathode decomposition",
        # "Anode decomposition heating [W.m-3]",
        # "Cathode decomposition heating [W.m-3]",
        # "SEI decomposition heating [W.m-3]",
        # "X-averaged Ohmic heating [W.m-3]",
        # "X-averaged irreversible electrochemical heating [W.m-3]",
        # "X-averaged total heating [W.m-3]",
        # "Negative electrode average extent of lithiation",
        
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
