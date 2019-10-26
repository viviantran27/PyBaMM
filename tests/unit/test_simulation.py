#
# Test the simulation class
#
import pybamm
import numpy as np
import unittest


class TestSimulation(unittest.TestCase):
    """Test the simulation class."""

    def test_simulation_setup(self):
        model = pybamm.lithium_ion.DFN()

        # test default setup
        sim = pybamm.Simulation(model)
        self.assertEqual(sim.parameter_values, model.default_parameter_values)
        self.assertEqual(sim.discretisation, model.default_discretisation)
        self.assertEqual(sim.solver, model.default_solver)
        self.assertEqual(str(sim), "Simulation for Doyle-Fuller-Newman model")

        # test custom setup
        parameter_values = pybamm.ParameterValues(
            chemistry=pybamm.parameter_sets.Marquis2019
        )
        geometry = pybamm.Geometry("2+1D macro", "1+1D micro")
        submesh_types = {
            "negative electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "separator": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive electrode": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "negative particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
            "positive particle": pybamm.MeshGenerator(pybamm.Uniform1DSubMesh),
        }
        var = pybamm.standard_spatial_vars
        var_pts = {
            var.x_n: 10,
            var.x_s: 10,
            var.x_p: 10,
            var.r_n: 5,
            var.r_p: 5,
            var.y: 10,
            var.z: 10,
        }
        spatial_methods = {
            "macroscale": pybamm.FiniteVolume,
            "negative particle": pybamm.FiniteVolume,
            "positive particle": pybamm.FiniteVolume,
        }
        solver = pybamm.ScipySolver(method="RK45")

        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            geometry=geometry,
            submesh_types=submesh_types,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
            name="test name",
        )
        self.assertEqual(sim.parameter_values, parameter_values)
        self.assertEqual(sim.geometry, geometry)
        self.assertEqual(sim.solver, solver)
        self.assertEqual(str(sim), "test name")

    def test_simulation_run(self):
        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model)
        solution = sim.run(np.linspace(0, 0.17))
        self.assertIsInstance(solution, pybamm.Solution)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
