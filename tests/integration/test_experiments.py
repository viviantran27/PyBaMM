#
# Test some experiments
#
import pybamm
import numpy as np
import unittest


class TestExperiments(unittest.TestCase):
    def test_discharge_rest_charge(self):
        experiment = pybamm.Experiment(
            [
                "Discharge at C/2 for 1 hour",
                "Rest for 1 hour",
                "Charge at C/2 for 1 hour",
            ],
            period="0.25 hours",
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model, experiment=experiment, solver=pybamm.CasadiSolver()
        )
        sim.solve()
        np.testing.assert_array_almost_equal(
            sim._solution["Time [h]"].entries, np.linspace(0, 3, 13)
        )
        cap = model.default_parameter_values["Cell capacity [A.h]"]
        np.testing.assert_array_almost_equal(
            sim._solution["Current [A]"].entries,
            [cap / 2] * 5 + [0] * 4 + [-cap / 2] * 4,
        )

    def test_gitt(self):
        experiment = pybamm.Experiment(
            ["Discharge at C/20 for 1 hour", "Rest for 1 hour"] * 10,
            period="6 minutes",
        )
        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(
            model, experiment=experiment, solver=pybamm.CasadiSolver()
        )
        sim.solve()
        np.testing.assert_array_almost_equal(
            sim._solution["Time [h]"].entries, np.arange(0, 20.01, 0.1)
        )
        cap = model.default_parameter_values["Cell capacity [A.h]"]
        np.testing.assert_array_almost_equal(
            sim._solution["Current [A]"].entries,
            [cap / 20] * 11 + [0] * 10 + ([cap / 20] * 10 + [0] * 10) * 9,
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
