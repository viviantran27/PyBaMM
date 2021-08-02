import pandas 
import numpy as np
import pybamm

# temperature_profile  = pandas.read_csv("pybamm/input/drive_cycles/ESC_100SOC_temperature.csv")
# timescale = 60
# temperature_interpolant = pybamm.Interpolant(temperature_profile[:][0], temperature_profile[:][1], timescale * pybamm.t)     
# t_eval = np.linspace(0,10, 10)
# for i in np.arange(1, len(t_eval) - 1): # define temperature change with time
#     dt = t_eval[i + 1] - t_eval[i]
#     print(T_av_dim.iloc[i][1])

# model_options = {"thermal": "x-lumped", "external submodels": ["thermal"]}
# model = pybamm.lithium_ion.SPMe(model_options)
# sim = pybamm.Simulation(model)
# t_eval = np.linspace(0, 100, 100)
# T_av = 0
# for i in np.arange(1, len(t_eval) - 1):
#     dt = t_eval[i + 1] - t_eval[i]
#     external_variables = {"Volume-averaged cell temperature": T_av}
#     T_av += 1
#     sim.step(dt, external_variables=external_variables)
# V = sim.solution["Terminal voltage [V]"].data
# T = sim.solution["Volume-averaged cell temperature [K]"].data


model_options = {
    "current collector": "potential pair",
    "dimensionality": 2,
    "external submodels": ["current collector"],
}
model = pybamm.lithium_ion.DFN(model_options)
yz_pts = 3
var_pts = {
    pybamm.standard_spatial_vars.x_n: 4,
    pybamm.standard_spatial_vars.x_s: 4,
    pybamm.standard_spatial_vars.x_p: 4,
    pybamm.standard_spatial_vars.r_n: 4,
    pybamm.standard_spatial_vars.r_p: 4,
    pybamm.standard_spatial_vars.y: yz_pts,
    pybamm.standard_spatial_vars.z: yz_pts,
}
sim = pybamm.Simulation(model, var_pts=var_pts)

# Simulate 100 seconds
t_eval = np.linspace(0, 100, 3)

for i in np.arange(1, len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]

    # provide phi_s_n and i_cc
    phi_s_n = np.zeros((yz_pts ** 2, 1))
    i_boundary_cc = np.ones((yz_pts ** 2, 1))
    external_variables = {
        "Negative current collector potential": phi_s_n,
        "Current collector current density": i_boundary_cc,
    }

    sim.step(dt, external_variables=external_variables)

    # obtain phi_s_n from the pybamm solution at the current time
    phi_s_p = sim.solution["Positive current collector potential"].data[:, -1]

output_variables =[
    # "Electrolyte concentration",
    # "Electrolyte potential [V]",
    # "Negative electrode potential [V]",
    "Current [A]",
    # "Interfacial current density",
    # "Volume-averaged cell temperature [K]",
    # "Cell temperature [K]",
    "Terminal voltage [V]"]

sim.plot(output_variables)