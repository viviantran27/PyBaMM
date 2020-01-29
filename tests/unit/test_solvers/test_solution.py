#
# Tests for the Solution class
#
import pybamm
import unittest
import numpy as np


class TestSolution(unittest.TestCase):
    def test_init(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        sol = pybamm.Solution(t, y)
        np.testing.assert_array_equal(sol.t, t)
        np.testing.assert_array_equal(sol.y, y)
        self.assertEqual(sol.t_event, None)
        self.assertEqual(sol.y_event, None)
        self.assertEqual(sol.termination, "final time")
        self.assertEqual(sol.inputs, {})
        self.assertEqual(sol.model, None)

    def test_append(self):
        # Set up first solution
        t1 = np.linspace(0, 1)
        y1 = np.tile(t1, (20, 1))
        sol1 = pybamm.Solution(t1, y1)
        sol1.solve_time = 1.5
        sol1.inputs = {}

        # Set up second solution
        t2 = np.linspace(1, 2)
        y2 = np.tile(t2, (20, 1))
        sol2 = pybamm.Solution(t2, y2)
        sol2.solve_time = 1
        sol1.append(sol2)

        # Test
        self.assertEqual(sol1.solve_time, 2.5)
        np.testing.assert_array_equal(sol1.t, np.concatenate([t1, t2[1:]]))
        np.testing.assert_array_equal(sol1.y, np.concatenate([y1, y2[:, 1:]], axis=1))

    def test_total_time(self):
        sol = pybamm.Solution([], None)
        sol.set_up_time = 0.5
        sol.solve_time = 1.2
        self.assertEqual(sol.total_time, 1.7)

    def test_getitem(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c
        model.variables["2c"] = 2 * c

        disc = pybamm.Discretisation()
        disc.process_model(model)
        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        # test create a new processed variable
        c_sol = solution["c"]
        self.assertIsInstance(c_sol, pybamm.ProcessedVariable)
        np.testing.assert_array_equal(c_sol.entries, c_sol(solution.t))

        # test call an already created variable
        solution.update("2c")
        twoc_sol = solution["2c"]
        self.assertIsInstance(twoc_sol, pybamm.ProcessedVariable)
        np.testing.assert_array_equal(twoc_sol.entries, twoc_sol(solution.t))
        np.testing.assert_array_equal(twoc_sol.entries, 2 * c_sol.entries)

    def test_save(self):
        model = pybamm.BaseModel()
        c = pybamm.Variable("c")
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables["c"] = c

        disc = pybamm.Discretisation()
        disc.process_model(model)
        solution = pybamm.ScipySolver().solve(model, np.linspace(0, 1))

        # test save data
        with self.assertRaises(ValueError):
            solution.save_data("test.pickle")
        # set variables first then save
        solution.update(["c"])
        solution.save_data("test.pickle")
        data_load = pybamm.load("test.pickle")
        np.testing.assert_array_equal(solution.data["c"], data_load["c"])

        # test save
        solution.save("test.pickle")
        solution_load = pybamm.load("test.pickle")
        self.assertEqual(solution.model.name, solution_load.model.name)
        np.testing.assert_array_equal(solution["c"].entries, solution_load["c"].entries)

    def test_solution_evals_with_inputs(self):
        model = pybamm.lithium_ion.SPM()
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.update({"Electrode height [m]": "[input]"})
        param.process_model(model)
        param.process_geometry(geometry)
        var = pybamm.standard_spatial_vars
        var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10}
        spatial_methods = model.default_spatial_methods
        solver = model.default_solver
        sim = pybamm.Simulation(
            model=model,
            geometry=geometry,
            parameter_values=param,
            var_pts=var_pts,
            spatial_methods=spatial_methods,
            solver=solver,
        )
        inputs = {"Electrode height [m]": 0.1}
        sim.solve(t_eval=np.linspace(0, 0.01, 10), inputs=inputs)
        time = sim.solution["Time [h]"](sim.solution.t)
        self.assertEqual(len(time), 10)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
