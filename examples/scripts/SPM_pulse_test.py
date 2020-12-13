#
# Perform a pulse test with the SPM Li-ion cell. Format data and heading 
# to acceptable csv for Matlab to extract RC parameters.
#
# NOTE: For solver integration error, reduce the t_eval endtime
# .
import pybamm
import numpy as np

pybamm.set_logging_level("INFO")

# load model
def pulse_test(pulse_time, rest_time, pulse_current):
    def current(t):
        floor = pybamm.Function(np.floor, t/(pulse_time + rest_time))
        mod_t = t-(pulse_time + rest_time)*floor
        pulse_signal = mod_t < pulse_time
        return pulse_signal * pulse_current
    return current

operating_mode = "current"

options = {
    "thermal": "x-lumped",
    "side reactions": "decomposition",
    "operating mode": operating_mode, 
}
models = [
    pybamm.lithium_ion.SPM({"thermal": "x-lumped","operating mode": operating_mode}, name="without decomposition"),
]

solutions = []
for model in models:
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
        # "Cell capacity [A.h]": 10, #match Kriston et al.
        # "Typical current [A]": 10, #match Kriston et al.
        # "Edge heat transfer coefficient [W.m-2.K-1]": 30,
        "Current function [A]": pulse_test(2*60, 5*60, 5*2),
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
    t_eval = np.linspace(0,3600*2, 36000)
    solution = model.default_solver.solve(model, t_eval)

    solutions.append(solution)

# plot
plot = pybamm.QuickPlot(
    solutions,
    [
        "X-averaged negative particle concentration",
        "X-averaged positive particle concentration",
        "Electrolyte concentration [mol.m-3]",
        "Current [A]",
        "Negative electrode potential [V]",
        "Electrolyte potential [V]",
        "Positive electrode potential [V]",
        "Terminal voltage [V]",
        "X-averaged cell temperature [K]",
        "X-averaged negative electrode extent of lithiation",
        
    ],
    time_unit="seconds",
    spatial_unit="um",
)
plot.dynamic_plot()
solution.save_data(
    "pulse.csv",
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "X-averaged cell temperature [K]",
    ],
    to_format="csv",
)
