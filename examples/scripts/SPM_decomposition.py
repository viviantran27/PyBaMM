#
# Example of SPM with decomposition reactions at high temperatures
#
# NOTE: For solver integration error, reduce the t_eval endtime
# .
import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
class ExternalCircuitFunction:

    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return V / I - pybamm.FunctionParameter("Function", {"Time [s]": pybamm.t})

options = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    "operating mode": ExternalCircuitFunction()
}
models = [
    pybamm.lithium_ion.SPM(options, name="with decomposition"),
    pybamm.lithium_ion.SPM({"thermal": "x-lumped","operating mode": ExternalCircuitFunction()}, name="without decomposition"),
]

solutions = []
for model in models:
    # turn off all events
    model.events = []
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
        "Ambient temperature [K]": 25 + 273, #180 + 273,
        "Initial temperature [K]": 25 + 273, #180 + 273,
        # "Function": 0.008},
        "Function": 0.01},
        check_already_exists=False,
    )
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

    # discretise model
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    # solve model for 1 hour
    t_eval = np.linspace(0, 14, 100)
    solution = model.default_solver.solve(model, t_eval)
    solutions.append(solution)

# plot
plot = pybamm.QuickPlot(
    solutions,
    [
        "X-averaged negative particle concentration [mol.m-3]",
        "X-averaged positive particle concentration [mol.m-3]",
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
        "Ambient temperature [K]",
        "Relative SEI thickness",
        "Fraction of Li in SEI",
        "Degree of conversion of cathode decomposition",
        "Anode decomposition heating [W.m-3]",
        "Cathode decomposition heating [W.m-3]",
        "SEI decomposition heating [W.m-3]",
        "X-averaged total heating [W.m-3]",
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
