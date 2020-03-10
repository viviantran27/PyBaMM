#
# Compares the full and lumped thermal models for a single layer Li-ion cell
#

import pybamm
import numpy as np

# load model
pybamm.set_logging_level("INFO")
class ExternalCircuitFunction:
    num_switches = 0

    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return V / I - pybamm.FunctionParameter("Function", pybamm.t)


options = {
    "thermal": "x-lumped",
    # "external submodels": ["thermal"],
    "operating mode": ExternalCircuitFunction()
}
model = pybamm.lithium_ion.DFN(options)
model.events = {}
parameter_values = model.default_parameter_values
parameter_values.update({"Function": 0.5}, check_already_exists=False)


# set external thermal model
sim = pybamm.Simulation(model, parameter_values=parameter_values)
param = pybamm.standard_parameters_lithium_ion
T_ref = parameter_values.evaluate(param.T_ref)
Delta_T = parameter_values.evaluate(param.Delta_T)
T_av_dim = 300
t_end = 301 #s
t_eval = np.linspace(0, t_end, 100) #in s

# define temperature change with time
for i in np.arange(1, len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]
    T_av = (T_av_dim - T_ref) / Delta_T
    external_variables = {"X-averaged cell temperature": T_av}
    T_av_dim += 0.5 #update T function 
    sim.step(dt, external_variables=external_variables)

#plot
output_variables =[
    "Electrolyte concentration",
    "Electrolyte potential [V]",
    "Negative electrode potential [V]",
    "Current [A]",
    "Interfacial current density",
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
    "Terminal voltage [V]"]

sim.plot(output_variables)