#
# Tests for the Base Parameter Values class
#
import pybamm
import os
import numpy as np

import unittest
import tests.shared as shared


class TestParameterValues(unittest.TestCase):
    def test_read_parameters_csv(self):
        data = pybamm.ParameterValues({}).read_parameters_csv(
            pybamm.get_parameters_filepath(
                os.path.join(
                    "input", "parameters",
                    "lithium-ion", "cathodes",
                    "lico2_Marquis2019", "parameters.csv"
                )
            )
        )
        self.assertEqual(data["Reference temperature [K]"], "298.15")

    def test_init(self):
        # from dict
        param = pybamm.ParameterValues({"a": 1})
        self.assertEqual(param["a"], 1)
        self.assertEqual(list(param.keys())[0], "a")
        self.assertEqual(list(param.values())[0], 1)
        self.assertEqual(list(param.items())[0], ("a", 1))

        # from file
        param = pybamm.ParameterValues(
            values=pybamm.get_parameters_filepath(
                "input/parameters/lithium-ion/cathodes/lico2_Marquis2019/"
                + "parameters.csv",
            )
        )
        self.assertEqual(param["Reference temperature [K]"], 298.15)

        # values vs chemistry
        with self.assertRaisesRegex(
            ValueError, "values and chemistry cannot both be None"
        ):
            pybamm.ParameterValues()
        with self.assertRaisesRegex(
            ValueError, "Only one of values and chemistry can be provided."
        ):
            pybamm.ParameterValues(values=1, chemistry={})

    def test_update_from_chemistry(self):
        # incomplete chemistry
        with self.assertRaisesRegex(KeyError, "must provide 'cell' parameters"):
            pybamm.ParameterValues(chemistry={"chemistry": "lithium-ion"})

    def test_update(self):
        param = pybamm.ParameterValues({"a": 1})
        self.assertEqual(param["a"], 1)
        # no conflict
        param.update({"a": 2})
        self.assertEqual(param["a"], 2)
        param.update({"a": 2}, check_conflict=True)
        self.assertEqual(param["a"], 2)
        # with conflict
        param.update({"a": 3})
        # via __setitem__
        param["a"] = 2
        self.assertEqual(param["a"], 2)
        with self.assertRaisesRegex(
            ValueError, "parameter 'a' already defined with value '2'"
        ):
            param.update({"a": 4}, check_conflict=True)
        # with parameter not existing yet
        with self.assertRaisesRegex(KeyError, "Cannot update parameter"):
            param.update({"b": 1})

    def test_check_and_update_parameter_values(self):
        # Can't provide a current density of 0, as this will cause a ZeroDivision error
        bad_values = {"Typical current [A]": 0}
        with self.assertRaisesRegex(ValueError, "Typical current"):
            pybamm.ParameterValues(bad_values)
        # can't provide both C-rate and current function
        bad_values = {"C-rate": 1, "Current function [A]": 5}
        with self.assertRaisesRegex(ValueError, "Cannot provide both"):
            pybamm.ParameterValues(bad_values)
        # if only C-rate and capacity provided, update current
        values = {"C-rate": 1, "Cell capacity [A.h]": 10}
        param = pybamm.ParameterValues(values)
        self.assertEqual(param["Current function [A]"], 10)
        # if only current and capacity provided, update C-rate
        values = {"Current function [A]": 1, "Cell capacity [A.h]": 10}
        param = pybamm.ParameterValues(values)
        self.assertEqual(param["C-rate"], 1 / 10)

        # With functions
        # if only C-rate and capacity provided, update current
        values = {"C-rate": pybamm.sin, "Cell capacity [A.h]": 10}
        param = pybamm.ParameterValues(values)
        self.assertEqual(param["Current function [A]"](2).evaluate(), 10 * np.sin(2))
        # if only current and capacity provided, update C-rate
        values = {"Current function [A]": pybamm.exp, "Cell capacity [A.h]": 10}
        param = pybamm.ParameterValues(values)
        self.assertEqual(param["C-rate"](5).evaluate(), np.exp(5) / 10)

        # With data
        # if only C-rate and capacity provided, update current
        x = np.linspace(0, 10)[:, np.newaxis]
        linear = np.hstack([x, 2 * x])
        values = {"C-rate": ("linear", linear), "Cell capacity [A.h]": 10}
        param = pybamm.ParameterValues(values)
        self.assertEqual(param["Current function [A]"][0], "linear_to_current")
        np.testing.assert_array_equal(
            param["Current function [A]"][1], np.hstack([x, 20 * x])
        )
        # if only current and capacity provided, update C-rate
        x = np.linspace(0, 10)[:, np.newaxis]
        linear = np.hstack([x, 2 * x])
        values = {"Current function [A]": ("linear", linear), "Cell capacity [A.h]": 10}
        param = pybamm.ParameterValues(values)
        self.assertEqual(param["C-rate"][0], "linear_to_Crate")
        np.testing.assert_array_almost_equal(
            param["C-rate"][1], np.hstack([x, 0.2 * x])
        )

        # With input parameters
        # if only C-rate and capacity provided, update current
        values = {"C-rate": "[input]", "Cell capacity [A.h]": 10}
        param = pybamm.ParameterValues(values)
        self.assertEqual(param["Current function [A]"](2).evaluate(u={"C-rate": 1}), 10)
        # if only current and capacity provided, update C-rate
        values = {"Current function [A]": "[input]", "Cell capacity [A.h]": 10}
        param = pybamm.ParameterValues(values)
        self.assertEqual(
            param["C-rate"](5).evaluate(u={"Current function [A]": 5}), 0.5
        )

    def test_process_symbol(self):
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2, "c": 3})
        # process parameter
        a = pybamm.Parameter("a")
        processed_a = parameter_values.process_symbol(a)
        self.assertIsInstance(processed_a, pybamm.Scalar)
        self.assertEqual(processed_a.value, 1)

        # process binary operation
        b = pybamm.Parameter("b")
        add = a + b
        processed_add = parameter_values.process_symbol(add)
        self.assertIsInstance(processed_add, pybamm.Addition)
        self.assertIsInstance(processed_add.children[0], pybamm.Scalar)
        self.assertIsInstance(processed_add.children[1], pybamm.Scalar)
        self.assertEqual(processed_add.children[0].value, 1)
        self.assertEqual(processed_add.children[1].value, 2)

        scal = pybamm.Scalar(34)
        mul = a * scal
        processed_mul = parameter_values.process_symbol(mul)
        self.assertIsInstance(processed_mul, pybamm.Multiplication)
        self.assertIsInstance(processed_mul.children[0], pybamm.Scalar)
        self.assertIsInstance(processed_mul.children[1], pybamm.Scalar)
        self.assertEqual(processed_mul.children[0].value, 1)
        self.assertEqual(processed_mul.children[1].value, 34)

        # process integral
        aa = pybamm.Parameter("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        integ = pybamm.Integral(aa, x)
        processed_integ = parameter_values.process_symbol(integ)
        self.assertIsInstance(processed_integ, pybamm.Integral)
        self.assertIsInstance(processed_integ.children[0], pybamm.Scalar)
        self.assertEqual(processed_integ.children[0].value, 1)
        self.assertEqual(processed_integ.integration_variable[0].id, x.id)

        # process unary operation
        grad = pybamm.Gradient(a)
        processed_grad = parameter_values.process_symbol(grad)
        self.assertIsInstance(processed_grad, pybamm.Gradient)
        self.assertIsInstance(processed_grad.children[0], pybamm.Scalar)
        self.assertEqual(processed_grad.children[0].value, 1)

        # process delta function
        aa = pybamm.Parameter("a")
        delta_aa = pybamm.DeltaFunction(aa, "left", "some domain")
        processed_delta_aa = parameter_values.process_symbol(delta_aa)
        self.assertIsInstance(processed_delta_aa, pybamm.DeltaFunction)
        self.assertEqual(processed_delta_aa.side, "left")
        processed_a = processed_delta_aa.children[0]
        self.assertIsInstance(processed_a, pybamm.Scalar)
        self.assertEqual(processed_a.value, 1)

        # process boundary operator (test for BoundaryValue)
        aa = pybamm.Parameter("a", domain=["negative electrode"])
        x = pybamm.SpatialVariable("x", domain=["negative electrode"])
        boundary_op = pybamm.BoundaryValue(aa * x, "left")
        processed_boundary_op = parameter_values.process_symbol(boundary_op)
        self.assertIsInstance(processed_boundary_op, pybamm.BoundaryOperator)
        processed_a = processed_boundary_op.children[0].children[0]
        processed_x = processed_boundary_op.children[0].children[1]
        self.assertIsInstance(processed_a, pybamm.Scalar)
        self.assertEqual(processed_a.value, 1)
        self.assertEqual(processed_x.id, x.id)

        # process broadcast
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        broad = pybamm.PrimaryBroadcast(a, whole_cell)
        processed_broad = parameter_values.process_symbol(broad)
        self.assertIsInstance(processed_broad, pybamm.Broadcast)
        self.assertEqual(processed_broad.domain, whole_cell)
        self.assertIsInstance(processed_broad.children[0], pybamm.Scalar)
        self.assertEqual(processed_broad.children[0].evaluate(), np.array([1]))

        # process concatenation
        conc = pybamm.Concatenation(
            pybamm.Vector(np.ones(10)), pybamm.Vector(2 * np.ones(15))
        )
        processed_conc = parameter_values.process_symbol(conc)
        self.assertIsInstance(processed_conc.children[0], pybamm.Vector)
        self.assertIsInstance(processed_conc.children[1], pybamm.Vector)
        np.testing.assert_array_equal(processed_conc.children[0].entries, 1)
        np.testing.assert_array_equal(processed_conc.children[1].entries, 2)

        # process domain concatenation
        c_e_n = pybamm.Variable("c_e_n", ["negative electrode"])
        c_e_s = pybamm.Variable("c_e_p", ["separator"])
        test_mesh = shared.get_mesh_for_testing()
        dom_con = pybamm.DomainConcatenation([a * c_e_n, b * c_e_s], test_mesh)
        processed_dom_con = parameter_values.process_symbol(dom_con)
        a_proc = processed_dom_con.children[0].children[0]
        b_proc = processed_dom_con.children[1].children[0]
        self.assertIsInstance(a_proc, pybamm.Scalar)
        self.assertIsInstance(b_proc, pybamm.Scalar)
        self.assertEqual(a_proc.value, 1)
        self.assertEqual(b_proc.value, 2)

        # process variable
        c = pybamm.Variable("c")
        processed_c = parameter_values.process_symbol(c)
        self.assertIsInstance(processed_c, pybamm.Variable)
        self.assertEqual(processed_c.name, "c")

        # process scalar
        d = pybamm.Scalar(14)
        processed_d = parameter_values.process_symbol(d)
        self.assertIsInstance(processed_d, pybamm.Scalar)
        self.assertEqual(processed_d.value, 14)

        # process array types
        e = pybamm.Vector(np.ones(4))
        processed_e = parameter_values.process_symbol(e)
        self.assertIsInstance(processed_e, pybamm.Vector)
        np.testing.assert_array_equal(processed_e.evaluate(), np.ones((4, 1)))

        f = pybamm.Matrix(np.ones((5, 6)))
        processed_f = parameter_values.process_symbol(f)
        self.assertIsInstance(processed_f, pybamm.Matrix)
        np.testing.assert_array_equal(processed_f.evaluate(), np.ones((5, 6)))

        # process statevector
        g = pybamm.StateVector(slice(0, 10))
        processed_g = parameter_values.process_symbol(g)
        self.assertIsInstance(processed_g, pybamm.StateVector)
        np.testing.assert_array_equal(
            processed_g.evaluate(y=np.ones(10)), np.ones((10, 1))
        )

        # not implemented
        sym = pybamm.Symbol("sym")
        with self.assertRaises(NotImplementedError):
            parameter_values.process_symbol(sym)

        # not found
        with self.assertRaises(KeyError):
            x = pybamm.Parameter("x")
            parameter_values.process_symbol(x)

    def test_process_input_parameter(self):
        parameter_values = pybamm.ParameterValues({"a": "[input]", "b": 3})
        # process input parameter
        a = pybamm.Parameter("a")
        processed_a = parameter_values.process_symbol(a)
        self.assertIsInstance(processed_a, pybamm.InputParameter)
        self.assertEqual(processed_a.evaluate(u={"a": 5}), 5)

        # process binary operation
        b = pybamm.Parameter("b")
        add = a + b
        processed_add = parameter_values.process_symbol(add)
        self.assertIsInstance(processed_add, pybamm.Addition)
        self.assertIsInstance(processed_add.children[0], pybamm.InputParameter)
        self.assertIsInstance(processed_add.children[1], pybamm.Scalar)
        self.assertEqual(processed_add.evaluate(u={"a": 4}), 7)

    def test_process_function_parameter(self):
        parameter_values = pybamm.ParameterValues(
            {
                "a": 3,
                "func": pybamm.load_function("process_symbol_test_function.py"),
                "const": 254,
            }
        )
        a = pybamm.InputParameter("a")

        # process function
        func = pybamm.FunctionParameter("func", a)
        processed_func = parameter_values.process_symbol(func)
        self.assertEqual(processed_func.evaluate(u={"a": 3}), 369)

        # process constant function
        const = pybamm.FunctionParameter("const", a)
        processed_const = parameter_values.process_symbol(const)
        self.assertIsInstance(processed_const, pybamm.Scalar)
        self.assertEqual(processed_const.evaluate(), 254)

        # process differentiated function parameter
        diff_func = func.diff(a)
        processed_diff_func = parameter_values.process_symbol(diff_func)
        self.assertEqual(processed_diff_func.evaluate(u={"a": 3}), 123)

        # function itself as input (different to the variable being an input)
        parameter_values = pybamm.ParameterValues({"func": "[input]"})
        a = pybamm.Scalar(3)
        func = pybamm.FunctionParameter("func", a)
        processed_func = parameter_values.process_symbol(func)
        self.assertEqual(processed_func.evaluate(u={"func": 13}), 13)

    def test_process_inline_function_parameters(self):
        def D(c):
            return c ** 2

        parameter_values = pybamm.ParameterValues({"Diffusivity": D})

        a = pybamm.InputParameter("a")
        func = pybamm.FunctionParameter("Diffusivity", a)

        processed_func = parameter_values.process_symbol(func)
        self.assertEqual(processed_func.evaluate(u={"a": 3}), 9)

        # process differentiated function parameter
        diff_func = func.diff(a)
        processed_diff_func = parameter_values.process_symbol(diff_func)
        self.assertEqual(processed_diff_func.evaluate(u={"a": 3}), 6)

    def test_multi_var_function_with_parameters(self):
        def D(a, b):
            return a * np.exp(b)

        parameter_values = pybamm.ParameterValues({"a": 3, "b": 0})
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        func = pybamm.Function(D, a, b)

        processed_func = parameter_values.process_symbol(func)
        self.assertIsInstance(processed_func, pybamm.Function)
        self.assertEqual(processed_func.evaluate(), 3)

    def test_multi_var_function_parameter(self):
        def D(a, b):
            return a * pybamm.exp(b)

        parameter_values = pybamm.ParameterValues({"a": 3, "b": 0, "Diffusivity": D})

        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        func = pybamm.FunctionParameter("Diffusivity", a, b)

        processed_func = parameter_values.process_symbol(func)
        self.assertEqual(processed_func.evaluate(), 3)

    def test_process_interpolant(self):
        x = np.linspace(0, 10)[:, np.newaxis]
        data = np.hstack([x, 2 * x])
        parameter_values = pybamm.ParameterValues(
            {"a": 3.01, "Diffusivity": ("times two", data)}
        )

        a = pybamm.Parameter("a")
        func = pybamm.FunctionParameter("Diffusivity", a)

        processed_func = parameter_values.process_symbol(func)
        self.assertIsInstance(processed_func, pybamm.Interpolant)
        self.assertEqual(processed_func.evaluate(), 6.02)

        # process differentiated function parameter
        diff_func = func.diff(a)
        processed_diff_func = parameter_values.process_symbol(diff_func)
        self.assertEqual(processed_diff_func.evaluate(), 2)

    def test_interpolant_against_function(self):
        parameter_values = pybamm.ParameterValues({})
        parameter_values.update(
            {
                "function": "[function]lico2_ocp_Dualfoil1998",
                "interpolation": "[data]lico2_data_example",
            },
            path=os.path.join(
                "input", "parameters", "lithium-ion", "cathodes", "lico2_Marquis2019",
            ),
            check_already_exists=False,
        )

        a = pybamm.InputParameter("a")
        func = pybamm.FunctionParameter("function", a)
        interp = pybamm.FunctionParameter("interpolation", a)

        processed_func = parameter_values.process_symbol(func)
        processed_interp = parameter_values.process_symbol(interp)
        np.testing.assert_array_almost_equal(
            processed_func.evaluate(u={"a": 0.6}),
            processed_interp.evaluate(u={"a": 0.6}),
            decimal=4,
        )

        # process differentiated function parameter
        diff_func = func.diff(a)
        diff_interp = interp.diff(a)
        processed_diff_func = parameter_values.process_symbol(diff_func)
        processed_diff_interp = parameter_values.process_symbol(diff_interp)
        np.testing.assert_array_almost_equal(
            processed_diff_func.evaluate(u={"a": 0.6}),
            processed_diff_interp.evaluate(u={"a": 0.6}),
            decimal=2,
        )

    def test_process_complex_expression(self):
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        par1 = pybamm.Parameter("par1")
        par2 = pybamm.Parameter("par2")
        scal1 = pybamm.Scalar(3)
        scal2 = pybamm.Scalar(4)
        expression = (scal1 * (par1 + var2)) / ((var1 - par2) + scal2)

        param = pybamm.ParameterValues(values={"par1": 1, "par2": 2})
        exp_param = param.process_symbol(expression)
        self.assertIsInstance(exp_param, pybamm.Division)
        # left side
        self.assertIsInstance(exp_param.children[0], pybamm.Multiplication)
        self.assertIsInstance(exp_param.children[0].children[0], pybamm.Scalar)
        self.assertIsInstance(exp_param.children[0].children[1], pybamm.Addition)
        self.assertTrue(
            isinstance(exp_param.children[0].children[1].children[0], pybamm.Scalar)
        )
        self.assertEqual(exp_param.children[0].children[1].children[0].value, 1)
        self.assertTrue(
            isinstance(exp_param.children[0].children[1].children[1], pybamm.Variable)
        )
        # right side
        self.assertIsInstance(exp_param.children[1], pybamm.Addition)
        self.assertTrue(
            isinstance(exp_param.children[1].children[0], pybamm.Subtraction)
        )
        self.assertTrue(
            isinstance(exp_param.children[1].children[0].children[0], pybamm.Variable)
        )
        self.assertTrue(
            isinstance(exp_param.children[1].children[0].children[1], pybamm.Scalar)
        )
        self.assertEqual(exp_param.children[1].children[0].children[1].value, 2)
        self.assertIsInstance(exp_param.children[1].children[1], pybamm.Scalar)

    def test_process_model(self):
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        var1 = pybamm.Variable("var1")
        var2 = pybamm.Variable("var2")
        model.rhs = {var1: a * pybamm.grad(var1)}
        model.algebraic = {var2: c * var2}
        model.initial_conditions = {var1: b, var2: d}
        model.boundary_conditions = {
            var1: {"left": (c, "Dirichlet"), "right": (d, "Neumann")}
        }
        model.variables = {
            "var1": var1,
            "var2": var2,
            "grad_var1": pybamm.grad(var1),
            "d_var1": d * var1,
        }
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2, "c": 3, "d": 42})
        parameter_values.process_model(model)
        # rhs
        self.assertIsInstance(model.rhs[var1], pybamm.Multiplication)
        self.assertIsInstance(model.rhs[var1].children[0], pybamm.Scalar)
        self.assertIsInstance(model.rhs[var1].children[1], pybamm.Gradient)
        self.assertEqual(model.rhs[var1].children[0].value, 1)
        # algebraic
        self.assertIsInstance(model.algebraic[var2], pybamm.Multiplication)
        self.assertIsInstance(model.algebraic[var2].children[0], pybamm.Scalar)
        self.assertIsInstance(model.algebraic[var2].children[1], pybamm.Variable)
        self.assertEqual(model.algebraic[var2].children[0].value, 3)
        # initial conditions
        self.assertIsInstance(model.initial_conditions[var1], pybamm.Scalar)
        self.assertEqual(model.initial_conditions[var1].value, 2)
        # boundary conditions
        bc_key = list(model.boundary_conditions.keys())[0]
        self.assertIsInstance(bc_key, pybamm.Variable)
        bc_value = list(model.boundary_conditions.values())[0]
        self.assertIsInstance(bc_value["left"][0], pybamm.Scalar)
        self.assertEqual(bc_value["left"][0].value, 3)
        self.assertIsInstance(bc_value["right"][0], pybamm.Scalar)
        self.assertEqual(bc_value["right"][0].value, 42)
        # variables
        self.assertEqual(model.variables["var1"].id, var1.id)
        self.assertIsInstance(model.variables["grad_var1"], pybamm.Gradient)
        self.assertTrue(
            isinstance(model.variables["grad_var1"].children[0], pybamm.Variable)
        )
        self.assertEqual(
            model.variables["d_var1"].id, (pybamm.Scalar(42, name="d") * var1).id
        )
        self.assertIsInstance(model.variables["d_var1"].children[0], pybamm.Scalar)
        self.assertTrue(
            isinstance(model.variables["d_var1"].children[1], pybamm.Variable)
        )

        # bad boundary conditions
        model = pybamm.BaseModel()
        model.algebraic = {var1: var1}
        x = pybamm.Parameter("x")
        model.boundary_conditions = {var1: {"left": (x, "Dirichlet")}}
        with self.assertRaises(KeyError):
            parameter_values.process_model(model)

    def test_update_model(self):
        param = pybamm.ParameterValues({})
        with self.assertRaises(NotImplementedError):
            param.update_model(None, None)

    def test_inplace(self):
        model = pybamm.lithium_ion.SPM()
        param = model.default_parameter_values
        new_model = param.process_model(model, inplace=False)

        for val in list(model.rhs.values()):
            self.assertTrue(val.has_symbol_of_classes(pybamm.Parameter))

        for val in list(new_model.rhs.values()):
            self.assertFalse(val.has_symbol_of_classes(pybamm.Parameter))

    def test_process_empty_model(self):
        model = pybamm.BaseModel()
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2, "c": 3, "d": 42})
        with self.assertRaisesRegex(
            pybamm.ModelError, "Cannot process parameters for empty model"
        ):
            parameter_values.process_model(model)

    def test_evaluate(self):
        parameter_values = pybamm.ParameterValues({"a": 1, "b": 2, "c": 3})
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        self.assertEqual(parameter_values.evaluate(a), 1)
        self.assertEqual(parameter_values.evaluate(a + (b * c)), 7)

        y = pybamm.StateVector(slice(0, 1))
        with self.assertRaises(ValueError):
            parameter_values.evaluate(y)
        array = pybamm.Array(np.array([1, 2, 3]))
        with self.assertRaises(ValueError):
            parameter_values.evaluate(array)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
