import pybamm
import unittest
import numpy as np


class TestQuickPlot(unittest.TestCase):
    def test_simple_ode_model(self):
        model = pybamm.BaseBatteryModel(name="Simple ODE Model")

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # Create variables: domain is explicitly empty since these variables are only
        # functions of time
        a = pybamm.Variable("a", domain=[])
        b = pybamm.Variable("b", domain=[])
        c = pybamm.Variable("c", domain=[])

        # Simple ODEs
        model.rhs = {a: pybamm.Scalar(2), b: pybamm.Scalar(0), c: -c}

        # Simple initial conditions
        model.initial_conditions = {
            a: pybamm.Scalar(0),
            b: pybamm.Scalar(1),
            c: pybamm.Scalar(1),
        }
        # no boundary conditions for an ODE model
        # Broadcast some of the variables
        model.variables = {
            "a": a,
            "b broadcasted": pybamm.FullBroadcast(b, whole_cell, "current collector"),
            "c broadcasted": pybamm.FullBroadcast(
                c, ["negative electrode", "separator"], "current collector"
            ),
        }

        # ODEs only (don't use jacobian)
        model.use_jacobian = False

        # Process and solve
        geometry = model.default_geometry
        param = model.default_parameter_values
        param.process_model(model)
        param.process_geometry(geometry)
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        solver = model.default_solver
        t_eval = np.linspace(0, 2, 100)
        solution = solver.solve(model, t_eval)
        quick_plot = pybamm.QuickPlot(solution)
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        quick_plot.axis.update({("a",): new_axis})
        self.assertEqual(quick_plot.axis[("a",)], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis[("a",)], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)

        # Test with different output variables
        quick_plot = pybamm.QuickPlot(solution, ["b broadcasted"])
        self.assertEqual(len(quick_plot.axis), 1)
        quick_plot.plot(0)

        quick_plot = pybamm.QuickPlot(
            solution, [["a", "a"], ["b broadcasted", "b broadcasted"], "c broadcasted"]
        )
        self.assertEqual(len(quick_plot.axis), 3)
        quick_plot.plot(0)

        # update the axis
        new_axis = [0, 0.5, 0, 1]
        var_key = ("c broadcasted",)
        quick_plot.axis.update({var_key: new_axis})
        self.assertEqual(quick_plot.axis[var_key], new_axis)

        # and now reset them
        quick_plot.reset_axis()
        self.assertNotEqual(quick_plot.axis[var_key], new_axis)

        # check dynamic plot loads
        quick_plot.dynamic_plot(testing=True)

        quick_plot.update(0.01)

        # Test longer name
        model.variables["Variable with a very long name"] = model.variables["a"]
        quick_plot = pybamm.QuickPlot(solution)
        quick_plot.plot(0)

        # Test errors
        with self.assertRaisesRegex(ValueError, "mismatching variable domains"):
            pybamm.QuickPlot(solution, [["a", "b broadcasted"]])
        model.variables["3D variable"] = disc.process_symbol(
            pybamm.FullBroadcast(
                1, "negative particle", {"secondary": "negative electrode"}
            )
        )
        with self.assertRaisesRegex(NotImplementedError, "cannot plot 3D variables"):
            pybamm.QuickPlot(solution, ["3D variable"])

    def test_loqs_spm_base(self):
        t_eval = np.linspace(0, 0.01, 2)

        # SPM
        for model in [pybamm.lithium_ion.SPM(), pybamm.lead_acid.LOQS()]:
            geometry = model.default_geometry
            param = model.default_parameter_values
            param.process_model(model)
            param.process_geometry(geometry)
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
            disc.process_model(model)
            solver = model.default_solver
            solution = solver.solve(model, t_eval)
            pybamm.QuickPlot(solution)

            # test quick plot of particle for spm
            if model.name == "Single Particle Model":
                output_variables = [
                    "X-averaged negative particle concentration [mol.m-3]",
                    "X-averaged positive particle concentration [mol.m-3]",
                ]
                pybamm.QuickPlot(solution, output_variables)

    def test_failure(self):
        with self.assertRaisesRegex(TypeError, "'solutions' must be"):
            pybamm.QuickPlot(1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
