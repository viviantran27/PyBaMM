#
# Tests for the base model class
#
import casadi
import pybamm
import numpy as np
import unittest
import os
import subprocess  # nosec
import platform


class TestBaseModel(unittest.TestCase):
    def test_rhs_set_get(self):
        model = pybamm.BaseModel()
        rhs = {
            pybamm.Symbol("c"): pybamm.Symbol("alpha"),
            pybamm.Symbol("d"): pybamm.Symbol("beta"),
        }
        model.rhs = rhs
        self.assertEqual(rhs, model.rhs)
        # test domains
        rhs = {
            pybamm.Symbol("c", domain=["negative electrode"]): pybamm.Symbol(
                "alpha", domain=["negative electrode"]
            ),
            pybamm.Symbol("d", domain=["positive electrode"]): pybamm.Symbol(
                "beta", domain=["positive electrode"]
            ),
        }
        model.rhs = rhs
        self.assertEqual(rhs, model.rhs)
        # non-matching domains should fail
        with self.assertRaises(pybamm.DomainError):
            model.rhs = {
                pybamm.Symbol("c", domain=["positive electrode"]): pybamm.Symbol(
                    "alpha", domain=["negative electrode"]
                )
            }

    def test_algebraic_set_get(self):
        model = pybamm.BaseModel()
        algebraic = {pybamm.Symbol("b"): pybamm.Symbol("c") - pybamm.Symbol("a")}
        model.algebraic = algebraic
        self.assertEqual(algebraic, model.algebraic)

    def test_initial_conditions_set_get(self):
        model = pybamm.BaseModel()
        initial_conditions = {
            pybamm.Symbol("c0"): pybamm.Symbol("gamma"),
            pybamm.Symbol("d0"): pybamm.Symbol("delta"),
        }
        model.initial_conditions = initial_conditions
        self.assertEqual(initial_conditions, model.initial_conditions)

        # Test number input
        c0 = pybamm.Symbol("c0")
        model.initial_conditions[c0] = 34
        self.assertIsInstance(model.initial_conditions[c0], pybamm.Scalar)
        self.assertEqual(model.initial_conditions[c0].value, 34)

        # Variable in initial conditions should fail
        with self.assertRaisesRegex(
            TypeError, "Initial conditions cannot contain 'Variable' objects"
        ):
            model.initial_conditions = {c0: pybamm.Variable("v")}

        # non-matching domains should fail
        with self.assertRaises(pybamm.DomainError):
            model.initial_conditions = {
                pybamm.Symbol("c", domain=["positive electrode"]): pybamm.Symbol(
                    "alpha", domain=["negative electrode"]
                )
            }

    def test_boundary_conditions_set_get(self):
        model = pybamm.BaseModel()
        boundary_conditions = {
            "c": {"left": ("epsilon", "Dirichlet"), "right": ("eta", "Dirichlet")}
        }
        model.boundary_conditions = boundary_conditions
        self.assertEqual(boundary_conditions, model.boundary_conditions)

        # Test number input
        c0 = pybamm.Symbol("c0")
        model.boundary_conditions[c0] = {
            "left": (-2, "Dirichlet"),
            "right": (4, "Dirichlet"),
        }
        self.assertIsInstance(model.boundary_conditions[c0]["left"][0], pybamm.Scalar)
        self.assertIsInstance(model.boundary_conditions[c0]["right"][0], pybamm.Scalar)
        self.assertEqual(model.boundary_conditions[c0]["left"][0].value, -2)
        self.assertEqual(model.boundary_conditions[c0]["right"][0].value, 4)
        self.assertEqual(model.boundary_conditions[c0]["left"][1], "Dirichlet")
        self.assertEqual(model.boundary_conditions[c0]["right"][1], "Dirichlet")

        # Check bad bc type
        bad_bcs = {c0: {"left": (-2, "bad type"), "right": (4, "bad type")}}
        with self.assertRaisesRegex(pybamm.ModelError, "boundary condition"):
            model.boundary_conditions = bad_bcs

    def test_variables_set_get(self):
        model = pybamm.BaseModel()
        variables = {"c": "alpha", "d": "beta"}
        model.variables = variables
        self.assertEqual(variables, model.variables)
        self.assertEqual(model.variable_names(), list(variables.keys()))

    def test_jac_set_get(self):
        model = pybamm.BaseModel()
        model.jacobian = "test"
        self.assertEqual(model.jacobian, "test")

    def test_model_dict_behaviour(self):
        model = pybamm.BaseModel()
        key = pybamm.Symbol("c")
        rhs = {key: pybamm.Symbol("alpha")}
        model.rhs = rhs
        self.assertEqual(model[key], rhs[key])
        self.assertEqual(model[key], model.rhs[key])

    def test_read_parameters(self):
        # Read parameters from different parts of the model
        model = pybamm.BaseModel()
        a = pybamm.Parameter("a")
        b = pybamm.Parameter("b")
        c = pybamm.Parameter("c")
        d = pybamm.Parameter("d")
        e = pybamm.Parameter("e")
        f = pybamm.Parameter("f")

        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -u * a}
        model.algebraic = {v: v - b}
        model.initial_conditions = {u: c, v: d}
        model.events = [pybamm.Event("u=e", u - e)]
        model.variables = {"v+f": v + f}

        self.assertEqual(
            set([x.name for x in model.parameters]),
            set([x.name for x in [a, b, c, d, e, f]]),
        )
        self.assertTrue(all(isinstance(x, pybamm.Parameter) for x in model.parameters))

    def test_read_input_parameters(self):
        # Read input parameters from different parts of the model
        model = pybamm.BaseModel()
        a = pybamm.InputParameter("a")
        b = pybamm.InputParameter("b")
        c = pybamm.InputParameter("c")
        d = pybamm.InputParameter("d")
        e = pybamm.InputParameter("e")
        f = pybamm.InputParameter("f")

        u = pybamm.Variable("u")
        v = pybamm.Variable("v")
        model.rhs = {u: -u * a}
        model.algebraic = {v: v - b}
        model.initial_conditions = {u: c, v: d}
        model.events = [pybamm.Event("u=e", u - e)]
        model.variables = {"v+f": v + f}

        self.assertEqual(
            set([x.name for x in model.input_parameters]),
            set([x.name for x in [a, b, c, d, e, f]]),
        )
        self.assertTrue(
            all(isinstance(x, pybamm.InputParameter) for x in model.input_parameters)
        )

    def test_update(self):
        # model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("c", domain=whole_cell)
        rhs = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        initial_conditions = {c: 1}
        boundary_conditions = {c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}}
        variables = {"c": c}
        model.rhs = rhs
        model.initial_conditions = initial_conditions
        model.boundary_conditions = boundary_conditions
        model.variables = variables

        # update with submodel
        submodel = pybamm.BaseModel()
        d = pybamm.Variable("d", domain=whole_cell)
        submodel.rhs = {
            d: 5 * pybamm.div(pybamm.grad(c)) + pybamm.div(pybamm.grad(d)) - 1
        }
        submodel.initial_conditions = {d: 3}
        submodel.boundary_conditions = {
            d: {"left": (4, "Dirichlet"), "right": (7, "Dirichlet")}
        }
        submodel.variables = {"d": d}
        model.update(submodel)

        # check
        self.assertEqual(model.rhs[d], submodel.rhs[d])
        self.assertEqual(model.initial_conditions[d], submodel.initial_conditions[d])
        self.assertEqual(model.boundary_conditions[d], submodel.boundary_conditions[d])
        self.assertEqual(model.variables["d"], submodel.variables["d"])
        self.assertEqual(model.rhs[c], rhs[c])
        self.assertEqual(model.initial_conditions[c], initial_conditions[c])
        self.assertEqual(model.boundary_conditions[c], boundary_conditions[c])
        self.assertEqual(model.variables["c"], variables["c"])

        # update with conflicting submodel
        submodel2 = pybamm.BaseModel()
        submodel2.rhs = {d: pybamm.div(pybamm.grad(d)) - 1}
        with self.assertRaises(pybamm.ModelError):
            model.update(submodel2)

        # update with multiple submodels
        submodel1 = submodel  # copy submodel from previous test
        submodel2 = pybamm.BaseModel()
        e = pybamm.Variable("e", domain=whole_cell)
        submodel2.rhs = {
            e: 5 * pybamm.div(pybamm.grad(d)) + pybamm.div(pybamm.grad(e)) - 1
        }
        submodel2.initial_conditions = {e: 3}
        submodel2.boundary_conditions = {
            e: {"left": (4, "Dirichlet"), "right": (7, "Dirichlet")}
        }

        model = pybamm.BaseModel()
        model.update(submodel1, submodel2)

        self.assertEqual(model.rhs[d], submodel1.rhs[d])
        self.assertEqual(model.initial_conditions[d], submodel1.initial_conditions[d])
        self.assertEqual(model.boundary_conditions[d], submodel1.boundary_conditions[d])
        self.assertEqual(model.rhs[e], submodel2.rhs[e])
        self.assertEqual(model.initial_conditions[e], submodel2.initial_conditions[e])
        self.assertEqual(model.boundary_conditions[e], submodel2.boundary_conditions[e])

    def test_new_copy(self):
        model = pybamm.BaseModel(name="a model")
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
        }
        model.use_jacobian = False
        model.use_simplify = False
        model.convert_to_format = "python"

        new_model = model.new_copy()
        self.assertEqual(new_model.name, model.name)
        self.assertEqual(new_model.use_jacobian, model.use_jacobian)
        self.assertEqual(new_model.use_simplify, model.use_simplify)
        self.assertEqual(new_model.convert_to_format, model.convert_to_format)
        self.assertEqual(new_model.timescale, model.timescale)

    def test_check_no_repeated_keys(self):
        model = pybamm.BaseModel()

        # rhs twice
        var = pybamm.Variable("var")
        model.rhs = {var: -1}
        var = pybamm.Variable("var")
        model.rhs.update({var: -1})
        with self.assertRaisesRegex(pybamm.ModelError, "Multiple equations specified"):
            model.check_no_repeated_keys()

        # rhs and algebraic
        model.rhs = {var: -1}
        var = pybamm.Variable("var")
        model.algebraic.update({var: var})
        with self.assertRaisesRegex(pybamm.ModelError, "Multiple equations specified"):
            model.check_no_repeated_keys()

    def test_check_well_posedness_variables(self):
        # Well-posed ODE model
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
        }
        model.check_well_posedness()

        # Well-posed DAE model
        e = pybamm.Variable("e", domain=whole_cell)
        model.algebraic = {e: e - c - d}
        model.check_well_posedness()

        # Underdetermined model - not enough differential equations
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1}
        model.algebraic = {e: e - c - d}
        with self.assertRaisesRegex(pybamm.ModelError, "underdetermined"):
            model.check_well_posedness()

        # Underdetermined model - not enough algebraic equations
        model.algebraic = {}
        with self.assertRaisesRegex(pybamm.ModelError, "underdetermined"):
            model.check_well_posedness()

        # Overdetermined model - repeated keys
        model.algebraic = {c: c - d, d: e + d}
        with self.assertRaisesRegex(pybamm.ModelError, "overdetermined"):
            model.check_well_posedness()
        # Overdetermined model - extra keys in algebraic
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -d}
        model.algebraic = {e: c - d}
        with self.assertRaisesRegex(pybamm.ModelError, "overdetermined"):
            model.check_well_posedness()
        model.rhs = {c: 1, d: -1}
        model.algebraic = {e: c - d}
        with self.assertRaisesRegex(pybamm.ModelError, "overdetermined"):
            model.check_well_posedness()

        # After discretisation, don't check for overdetermined from extra algebraic keys
        model = pybamm.BaseModel()
        model.algebraic = {c: 5 * pybamm.StateVector(slice(0, 15)) - 1}
        # passes with post_discretisation=True
        model.check_well_posedness(post_discretisation=True)
        # fails with post_discretisation=False (default)
        with self.assertRaisesRegex(pybamm.ModelError, "extra algebraic keys"):
            model.check_well_posedness()

        # before discretisation, fail if the algebraic eqn keys don't appear in the eqns
        model = pybamm.BaseModel()
        model.algebraic = {c: d - 2, d: d - c}
        with self.assertRaisesRegex(
            pybamm.ModelError,
            "each variable in the algebraic eqn keys must appear in the eqn",
        ):
            model.check_well_posedness()
        # passes when we switch the equations around
        model.algebraic = {c: d - c, d: d - 2}
        model.check_well_posedness()

        # after discretisation, algebraic equation without a StateVector fails
        model = pybamm.BaseModel()
        model.algebraic = {
            c: 1,
            d: pybamm.StateVector(slice(0, 15)) - pybamm.StateVector(slice(15, 30)),
        }
        with self.assertRaisesRegex(
            pybamm.ModelError,
            "each algebraic equation must contain at least one StateVector",
        ):
            model.check_well_posedness(post_discretisation=True)

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: d.diff(pybamm.t), d: -1}
        model.initial_conditions = {c: 1, d: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.algebraic = {c: 2 * d - c, d: c * d.diff(pybamm.t) - d}
        model.initial_conditions = {c: 1, d: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: d.diff(pybamm.t), d: -1}
        model.initial_conditions = {c: 1, d: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of variable found"
        ):
            model.check_well_posedness()

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.algebraic = {
            d: 5 * pybamm.StateVector(slice(0, 15)) - 1,
            c: 5 * pybamm.StateVectorDot(slice(0, 15)) - 1,
        }
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of state vector found"
        ):
            model.check_well_posedness(post_discretisation=True)

        # model must be in semi-explicit form
        model = pybamm.BaseModel()
        model.rhs = {c: 5 * pybamm.StateVectorDot(slice(0, 15)) - 1}
        model.initial_conditions = {c: 1}
        with self.assertRaisesRegex(
            pybamm.ModelError, "time derivative of state vector found"
        ):
            model.check_well_posedness(post_discretisation=True)

    def test_check_well_posedness_initial_boundary_conditions(self):
        # Well-posed model - Dirichlet
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        c = pybamm.Variable("c", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        model.initial_conditions = {c: 1}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.check_well_posedness()

        # Well-posed model - Neumann
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.check_well_posedness()

        # Model with bad initial conditions (expect assertion error)
        d = pybamm.Variable("d", domain=whole_cell)
        model.initial_conditions = {d: 3}
        with self.assertRaisesRegex(pybamm.ModelError, "initial condition"):
            model.check_well_posedness()

        # Model with bad boundary conditions - Dirichlet (expect assertion error)
        d = pybamm.Variable("d", domain=whole_cell)
        model.initial_conditions = {c: 3}
        model.boundary_conditions = {
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        with self.assertRaisesRegex(pybamm.ModelError, "boundary condition"):
            model.check_well_posedness()

        # Model with bad boundary conditions - Neumann (expect assertion error)
        d = pybamm.Variable("d", domain=whole_cell)
        model.initial_conditions = {c: 3}
        model.boundary_conditions = {
            d: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        with self.assertRaisesRegex(pybamm.ModelError, "boundary condition"):
            model.check_well_posedness()

        # Algebraic well-posed model
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        model = pybamm.BaseModel()
        model.algebraic = {c: 5 * pybamm.div(pybamm.grad(c)) - 1}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        model.check_well_posedness()
        model.boundary_conditions = {
            c: {"left": (0, "Neumann"), "right": (0, "Neumann")}
        }
        model.check_well_posedness()

        # Algebraic model with bad boundary conditions
        model.boundary_conditions = {
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")}
        }
        with self.assertRaisesRegex(pybamm.ModelError, "boundary condition"):
            model.check_well_posedness()

    def test_check_well_posedness_output_variables(self):
        model = pybamm.BaseModel()
        whole_cell = ["negative electrode", "separator", "positive electrode"]
        c = pybamm.Variable("c", domain=whole_cell)
        d = pybamm.Variable("d", domain=whole_cell)
        model.rhs = {c: 5 * pybamm.div(pybamm.grad(d)) - 1, d: -c}
        model.initial_conditions = {c: 1, d: 2}
        model.boundary_conditions = {
            c: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
            d: {"left": (0, "Dirichlet"), "right": (0, "Dirichlet")},
        }
        model._variables = {
            "something": None,
            "something else": c,
            "another thing": None,
        }

        # Check warning raised
        with self.assertWarns(pybamm.ModelWarning):
            model.check_well_posedness()

        # Check None entries have been removed from the variables dictionary
        for key, item in model._variables.items():
            self.assertIsNotNone(item)

        # check error raised if undefined variable in list of Variables
        pybamm.settings.debug_mode = True
        model = pybamm.BaseModel()
        model.rhs = {c: -c}
        model.initial_conditions = {c: 1}
        model.variables = {"d": d}
        with self.assertRaisesRegex(pybamm.ModelError, "No key set for variable"):
            model.check_well_posedness()

        # check error is raised even if some modified form of d is in model.rhs
        two_d = 2 * d
        model.rhs[two_d] = -d
        model.initial_conditions[two_d] = 1
        with self.assertRaisesRegex(pybamm.ModelError, "No key set for variable"):
            model.check_well_posedness()

        # add d to rhs, fine
        model.rhs[d] = -d
        model.initial_conditions[d] = 1
        model.check_well_posedness()

    def test_export_casadi(self):
        model = pybamm.BaseModel()
        t = pybamm.t
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {a: -a * p}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: q, b: 1}
        model.variables = {"a+b": a + b - t}

        out = model.export_casadi_objects(["a+b"])

        # Try making a function from the outputs
        t, x, z, p = out["t"], out["x"], out["z"], out["inputs"]
        x0, z0 = out["x0"], out["z0"]
        rhs, alg = out["rhs"], out["algebraic"]
        var = out["variables"]["a+b"]
        jac_rhs, jac_alg = out["jac_rhs"], out["jac_algebraic"]
        x0_fn = casadi.Function("x0", [p], [x0])
        z0_fn = casadi.Function("x0", [p], [z0])
        rhs_fn = casadi.Function("rhs", [t, x, z, p], [rhs])
        alg_fn = casadi.Function("alg", [t, x, z, p], [alg])
        jac_rhs_fn = casadi.Function("jac_rhs", [t, x, z, p], [jac_rhs])
        jac_alg_fn = casadi.Function("jac_alg", [t, x, z, p], [jac_alg])
        var_fn = casadi.Function("var", [t, x, z, p], [var])

        # Test that function values are as expected
        self.assertEqual(x0_fn([0, 5]), 5)
        self.assertEqual(z0_fn([0, 0]), 1)
        self.assertEqual(rhs_fn(0, 3, 2, [7, 2]), -21)
        self.assertEqual(alg_fn(0, 3, 2, [7, 2]), 1)
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [8, 9])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [8, 9])), [[1, -1]])
        self.assertEqual(var_fn(6, 3, 2, [7, 2]), -1)

        # Now change the order of input parameters
        out = model.export_casadi_objects(["a+b"], input_parameter_order=["q", "p"])

        # Try making a function from the outputs
        t, x, z, p = out["t"], out["x"], out["z"], out["inputs"]
        x0, z0 = out["x0"], out["z0"]
        rhs, alg = out["rhs"], out["algebraic"]
        var = out["variables"]["a+b"]
        jac_rhs, jac_alg = out["jac_rhs"], out["jac_algebraic"]
        x0_fn = casadi.Function("x0", [p], [x0])
        z0_fn = casadi.Function("x0", [p], [z0])
        rhs_fn = casadi.Function("rhs", [t, x, z, p], [rhs])
        alg_fn = casadi.Function("alg", [t, x, z, p], [alg])
        jac_rhs_fn = casadi.Function("jac_rhs", [t, x, z, p], [jac_rhs])
        jac_alg_fn = casadi.Function("jac_alg", [t, x, z, p], [jac_alg])
        var_fn = casadi.Function("var", [t, x, z, p], [var])

        # Test that function values are as expected
        self.assertEqual(x0_fn([5, 0]), 5)
        self.assertEqual(z0_fn([0, 0]), 1)
        self.assertEqual(rhs_fn(0, 3, 2, [2, 7]), -21)
        self.assertEqual(alg_fn(0, 3, 2, [2, 7]), 1)
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [9, 8])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [9, 8])), [[1, -1]])
        self.assertEqual(var_fn(6, 3, 2, [2, 7]), -1)

        # Test model with external variable runs
        model_options = {"thermal": "lumped", "external submodels": ["thermal"]}
        model = pybamm.lithium_ion.SPMe(model_options)
        sim = pybamm.Simulation(model)
        sim.build()
        variable_names = ["Volume-averaged cell temperature"]
        out = sim.built_model.export_casadi_objects(variable_names)

        # Test fails if not discretised
        with self.assertRaisesRegex(
            pybamm.DiscretisationError, "Cannot automatically discretise model"
        ):
            model.export_casadi_objects(["Electrolyte concentration"])

    @unittest.skipIf(platform.system() == "Windows", "Skipped for Windows")
    def test_generate_casadi(self):
        model = pybamm.BaseModel()
        t = pybamm.t
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        p = pybamm.InputParameter("p")
        q = pybamm.InputParameter("q")
        model.rhs = {a: -a * p}
        model.algebraic = {b: a - b}
        model.initial_conditions = {a: q, b: 1}
        model.variables = {"a+b": a + b - t}

        # Generate C code
        model.generate("test.c", ["a+b"])

        # Compile
        subprocess.run(["gcc", "-fPIC", "-shared", "-o", "test.so", "test.c"])  # nosec

        # Read the generated functions
        x0_fn = casadi.external("x0", "./test.so")
        z0_fn = casadi.external("z0", "./test.so")
        rhs_fn = casadi.external("rhs_", "./test.so")
        alg_fn = casadi.external("alg_", "./test.so")
        jac_rhs_fn = casadi.external("jac_rhs", "./test.so")
        jac_alg_fn = casadi.external("jac_alg", "./test.so")
        var_fn = casadi.external("variables", "./test.so")

        # Test that function values are as expected
        self.assertEqual(x0_fn([0, 5]), 5)
        self.assertEqual(z0_fn([0, 0]), 1)
        self.assertEqual(rhs_fn(0, 3, 2, [7, 2]), -21)
        self.assertEqual(alg_fn(0, 3, 2, [7, 2]), 1)
        np.testing.assert_array_equal(np.array(jac_rhs_fn(5, 6, 7, [8, 9])), [[-8, 0]])
        np.testing.assert_array_equal(np.array(jac_alg_fn(5, 6, 7, [8, 9])), [[1, -1]])
        self.assertEqual(var_fn(6, 3, 2, [7, 2]), -1)

        # Remove generated files.
        os.remove("test.c")
        os.remove("test.so")


class TestStandardBatteryBaseModel(unittest.TestCase):
    def test_default_solver(self):
        model = pybamm.BaseBatteryModel()
        self.assertIsInstance(
            model.default_solver, (pybamm.ScipySolver, pybamm.ScikitsOdeSolver)
        )

        # check that default_solver gives you a new solver, not an internal object
        solver = model.default_solver
        solver = pybamm.BaseModel()
        self.assertIsInstance(
            model.default_solver, (pybamm.ScipySolver, pybamm.ScikitsOdeSolver)
        )
        self.assertIsInstance(solver, pybamm.BaseModel)

        # check that adding algebraic variables gives DAE solver
        a = pybamm.Variable("a")
        model.algebraic = {a: a - 1}
        self.assertIsInstance(
            model.default_solver, (pybamm.IDAKLUSolver, pybamm.CasadiSolver)
        )

        # Check that turning off jacobian gives casadi solver
        model.use_jacobian = False
        self.assertIsInstance(model.default_solver, pybamm.CasadiSolver)

    def test_default_parameters(self):
        # check parameters are read in ok
        model = pybamm.BaseBatteryModel()
        self.assertEqual(
            model.default_parameter_values["Reference temperature [K]"], 298.15
        )

        # change path and try again

        cwd = os.getcwd()
        os.chdir("..")
        model = pybamm.BaseBatteryModel()
        self.assertEqual(
            model.default_parameter_values["Reference temperature [K]"], 298.15
        )
        os.chdir(cwd)

    def test_timescale(self):
        model = pybamm.BaseModel()
        self.assertEqual(model.timescale.evaluate(), 1)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
