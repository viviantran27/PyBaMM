import pybamm

# model = pybamm.lithium_ion.SPM()
model = pybamm.lead_acid.Full()

sim = pybamm.Simulation(model)
sim.solve()
sim.plot()
