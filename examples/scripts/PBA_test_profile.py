#
# Simulate drive cycle loaded from csv file
#
import pybamm

pybamm.set_logging_level("INFO")

# load model and update parameters so the input current is the PBA_discharge_test "drive cycle"
model = pybamm.lead_acid.Full({"thermal": "x-full"})
param = model.default_parameter_values
param.update({
    "Electrode height [m]":0.114*0.051,
    "Electrode width [m]":0.065,
    "Number of electrodes connected in parallel to make a cell":1, #8
    "Initial State of Charge":1.03,
    # "Positive electrode conductivity [S.m-1]":80000*0.1,
    # "Negative electrode conductivity [S.m-1]":4800000*0.1,
    "Negative electrode thickness [m]":0.0009*5,
    "Positive electrode thickness [m]":0.00125*5,
    # "Maximum porosity of negative electrode":0.53*0.95,
    # "Maximum porosity of positive electrode":0.53*0.95,
    # "Maximum porosity of separator":0.92*0.95,
    "Separator thickness [m]":0.0015*6,
    },
    check_already_exists=False
)
param["Current function [A]"] = "[current data]PBA_discharge_test"

# create and run simulation using the CasadiSolver
sim = pybamm.Simulation(
    model, parameter_values=param, solver=pybamm.CasadiSolver()
)
sim.solve()

solution = sim.solution
solution.save_data(
    "output.csv",
    [
        "Time [h]",
        "Current [A]",
        "Terminal voltage [V]",
        "X-averaged cell temperature [K]",
        "State of Charge"
    ],
    to_format="csv",
)
print("Done saving data to csv.")

output_variables = [
    "Terminal voltage [V]",
    "X-averaged cell temperature [K]",
    "Electrolyte concentration [Molar]",
    "Current [A]",
]
sim.plot(output_variables)

