#
# Standard basic tests for any model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import copy
import numpy as np


class StandardModelTest(object):
    """ Basic processing test for the models. """

    def __init__(self, model):
        self.model = model
        # Set default parameters
        self.parameter_values = model.default_parameter_values
        # Process geometry
        self.parameter_values.process_geometry(model.default_geometry)
        geometry = model.default_geometry
        # Set default discretisation
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        self.disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        # Set default solver
        self.solver = model.default_solver

    def test_processing_parameters(self, parameter_values=None):
        # Overwrite parameters if given
        if parameter_values is not None:
            self.parameter_values = parameter_values
        self.parameter_values.process_model(self.model)
        # Model should still be well-posed after processing
        self.model.check_well_posedness()
        # No Parameter or FunctionParameter nodes in the model
        for eqn in {**self.model.rhs, **self.model.algebraic}.values():
            if any(
                [
                    isinstance(x, (pybamm.Parameter, pybamm.FunctionParameter))
                    for x in eqn.pre_order()
                ]
            ):
                raise TypeError(
                    "Not all Parameter and FunctionParameter objects processed"
                )

    def test_processing_disc(self, disc=None):
        # Overwrite discretisation if given
        if disc is not None:
            self.disc = disc
        self.disc.process_model(self.model)

        # Model should still be well-posed after processing
        self.model.check_well_posedness(post_discretisation=True)

    def test_solving(self, solver=None, t_eval=None):
        # Overwrite solver if given
        if solver is not None:
            self.solver = solver
        if t_eval is None:
            t_eval = np.linspace(0, 1, 100)

        self.solver.solve(self.model, t_eval)

    def test_all(self, param=None, disc=None, solver=None, t_eval=None):
        self.model.check_well_posedness()
        self.test_processing_parameters(param)
        self.test_processing_disc(disc)
        self.test_solving(solver, t_eval)

    def test_update_parameters(self, param):
        # check if geometry has changed, throw error if so (need to re-discretise)
        if any(
            [
                length in param.keys()
                and param[length] != self.parameter_values[length]
                for length in [
                    "Negative electrode width",
                    "Separator width",
                    "Positive electrode width",
                ]
            ]
        ):
            raise ValueError(
                "geometry has changed, the orginal model needs to be re-discretised"
            )
        # otherwise update self.param and change the parameters in the discretised model
        self.param = param
        param.process_discretised_model(self.model, self.disc)
        # Model should still be well-posed after processing
        self.model.check_well_posedness()


class OptimisationsTest(object):
    """ Test that the optimised models give the same result as the original model. """

    def __init__(self, model, parameter_values=None, disc=None):
        # Set parameter values
        if parameter_values is None:
            parameter_values = model.default_parameter_values
        # Process model and geometry
        parameter_values.process_model(model)
        parameter_values.process_geometry(model.default_geometry)
        geometry = model.default_geometry
        # Set discretisation
        if disc is None:
            mesh = pybamm.Mesh(
                geometry, model.default_submesh_types, model.default_var_pts
            )
            disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        # Discretise model
        disc.process_model(model)

        self.model = model

    def evaluate_model(self, simplify=False, use_known_evals=False):
        result = np.array([])
        for eqn in [self.model.concatenated_rhs, self.model.concatenated_algebraic]:
            if eqn is not None:
                if simplify:
                    eqn = eqn.simplify()

                y = self.model.concatenated_initial_conditions
                if use_known_evals:
                    eqn_eval, known_evals = eqn.evaluate(0, y, known_evals={})
                else:
                    eqn_eval = eqn.evaluate(0, y)
            else:
                eqn_eval = np.array([])

            result = np.concatenate([result, eqn_eval])

        return result


def get_manufactured_solution_errors(model, has_spatial_derivatives=True):
    # Process model and geometry
    param = model.default_parameter_values
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    # Add manufactured solution to model
    ms = pybamm.ManufacturedSolution()
    ms.process_model(model)
    # Set up solver
    t_eval = np.linspace(0, 1)
    solver = model.default_solver

    # Function for convergence testing
    def get_l2_error(n):
        model_copy = copy.deepcopy(model)
        # Set up discretisation
        var = pybamm.standard_spatial_vars
        submesh_pts = {var.x_n: n, var.x_s: n, var.x_p: n, var.r_n: n, var.r_p: n}
        mesh = pybamm.Mesh(geometry, model_copy.default_submesh_types, submesh_pts)
        disc = pybamm.Discretisation(mesh, model_copy.default_spatial_methods)

        # Discretise and solve
        disc.process_model(model_copy)
        solver.solve(model_copy, t_eval)
        t, y = solver.t, solver.y
        # Process model and exact solutions
        approx_all = np.array([])
        exact_all = np.array([])
        for var_string, man_var in ms.man_var_strings.items():
            # Approximate solution from solving the model
            approx = pybamm.ProcessedVariable(
                model_copy.variables[var_string], t, y, mesh=disc.mesh
            ).entries
            approx_all = np.concatenate([approx_all, np.reshape(approx, -1)])
            # Exact solution from manufactured solution
            exact = disc.process_symbol(man_var).evaluate(t=t)
            exact_all = np.concatenate([exact_all, np.reshape(exact, -1)])

        # error
        import ipdb

        ipdb.set_trace()
        error = np.linalg.norm(approx_all - exact_all) / np.linalg.norm(exact_all)
        return error

    if has_spatial_derivatives:
        # Get errors
        ns = 10 * (2 ** np.arange(2, 7))
        return np.array([get_l2_error(int(n)) for n in ns])
    else:
        return get_l2_error(1)
