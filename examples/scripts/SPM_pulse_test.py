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
    "thermal": "two-state lumped",
    "side reactions": "decomposition",
    "operating mode": operating_mode, 
}
models = [
    pybamm.lithium_ion.SPM({"thermal": "lumped", "operating mode": operating_mode}, name="without decomposition"),
]

solutions = []
for model in models:
    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Cai2019)
    param.update(
        {
        # "Cell capacity [A.h]": 0.5, 
        # "Typical current [A]": 0.5,
        "Current function [A]": pulse_test(2*60, 5*60, 9),
        "Edge heat transfer coefficient [W.m-2.K-1]": 3000,
        "Negative electrode thickness [m]":62E-06*4.2/5, # cell 43 
        "Positive electrode thickness [m]":67E-06*4.2/5,
        # "Separator thickness [m]":12E-06,
        # "Positive electrode conductivity [S.m-1]":100,
        # "Negative electrode conductivity [S.m-1]":100,
        # "Positive particle radius [m]": 3.5E-06*2,
        # "Negative particle radius [m]":2.5E-06*2,
        # "Initial concentration in negative electrode [mol.m-3]": 0.87*28746, #x0 (soc_0*(0.87-0.0017)+0.0017)*28746 (0.0017) * Csmax_n(28746)
        # "Initial concentration in positive electrode [mol.m-3]": 0.025*35380, #y0 (0.8907-soc_0*(0.8907-0.03))*35380 (0.8907) * Csmax_p(35380) 
        "Ambient temperature [K]": 23+273.15,
        "Initial temperature [K]": 23+273.15,
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
    t_eval = np.linspace(0,5532, 3600)
    solution = model.default_solver.solve(model, t_eval)

    solutions.append(solution)

# save data to csv
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

