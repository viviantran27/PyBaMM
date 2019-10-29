#
# Base solver class
#
import casadi
import pybamm
import numpy as np
from scipy import optimize
from scipy.sparse import issparse


class DaeSolver(pybamm.BaseSolver):
    """Solve a discretised model.

    Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str, optional
        The method to use to find initial conditions (default is "lm")
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-6).
    max_steps: int, optional
        The maximum number of steps the solver will take before terminating
        (default is 1000).
    """

    def __init__(
        self,
        method=None,
        rtol=1e-6,
        atol=1e-6,
        root_method="lm",
        root_tol=1e-6,
        max_steps=1000,
    ):
        super().__init__(method, rtol, atol)
        self.root_method = root_method
        self.root_tol = root_tol
        self.max_steps = max_steps

    @property
    def root_method(self):
        return self._root_method

    @root_method.setter
    def root_method(self, method):
        self._root_method = method

    @property
    def root_tol(self):
        return self._root_tol

    @root_tol.setter
    def root_tol(self, tol):
        self._root_tol = tol

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, max_steps):
        self._max_steps = max_steps

    def compute_solution(self, model, t_eval):
        """Calculate the solution of the model at specified times.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        t_eval : numeric type
            The times at which to compute the solution

        """
        timer = pybamm.Timer()

        solve_start_time = timer.time()
        pybamm.logger.info("Calling DAE solver")
        solution = self.integrate(
            self.residuals,
            self.y0,
            t_eval,
            events=self.event_funs,
            mass_matrix=model.mass_matrix.entries,
            jacobian=self.jacobian,
        )
        solve_time = timer.time() - solve_start_time

        # Identify the event that caused termination
        termination = self.get_termination_reason(solution, self.events)

        return solution, solve_time, termination

    def set_up(self, model):
        """Unpack model, perform checks, simplify and calculate jacobian.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        """
        # create simplified rhs, algebraic and event expressions
        concatenated_rhs = model.concatenated_rhs
        concatenated_algebraic = model.concatenated_algebraic
        events = model.events

        if model.use_simplify:
            # set up simplification object, for re-use of dict
            simp = pybamm.Simplification()
            pybamm.logger.info("Simplifying RHS")
            concatenated_rhs = simp.simplify(concatenated_rhs)
            pybamm.logger.info("Simplifying algebraic")
            concatenated_algebraic = simp.simplify(concatenated_algebraic)
            pybamm.logger.info("Simplifying events")
            events = {name: simp.simplify(event) for name, event in events.items()}

        if model.use_jacobian:
            # Create Jacobian from concatenated rhs and algebraic
            y = pybamm.StateVector(
                slice(0, np.size(model.concatenated_initial_conditions))
            )
            # set up Jacobian object, for re-use of dict
            jacobian = pybamm.Jacobian()
            pybamm.logger.info("Calculating jacobian")
            jac_rhs = jacobian.jac(concatenated_rhs, y)
            jac_algebraic = jacobian.jac(concatenated_algebraic, y)
            jac = pybamm.SparseStack(jac_rhs, jac_algebraic)
            model.jacobian = jac
            model.jacobian_rhs = jac_rhs
            model.jacobian_algebraic = jac_algebraic

            if model.use_simplify:
                pybamm.logger.info("Simplifying jacobian")
                jac_algebraic = simp.simplify(jac_algebraic)
                jac = simp.simplify(jac)

            if model.convert_to_format == "python":
                pybamm.logger.info("Converting jacobian to python")
                jac_algebraic = pybamm.EvaluatorPython(jac_algebraic)
                jac = pybamm.EvaluatorPython(jac)

            def jacobian(t, y):
                return jac.evaluate(t, y, known_evals={})[0]

            def jacobian_alg(t, y):
                return jac_algebraic.evaluate(t, y, known_evals={})[0]

        else:
            jacobian = None
            jacobian_alg = None

        if model.convert_to_format == "python":
            pybamm.logger.info("Converting RHS to python")
            concatenated_rhs = pybamm.EvaluatorPython(concatenated_rhs)
            pybamm.logger.info("Converting algebraic to python")
            concatenated_algebraic = pybamm.EvaluatorPython(concatenated_algebraic)
            pybamm.logger.info("Converting events to python")
            events = {
                name: pybamm.EvaluatorPython(event) for name, event in events.items()
            }

        # Calculate consistent initial conditions for the algebraic equations
        def rhs(t, y):
            return concatenated_rhs.evaluate(t, y, known_evals={})[0][:, 0]

        def algebraic(t, y):
            return concatenated_algebraic.evaluate(t, y, known_evals={})[0][:, 0]

        if len(model.algebraic) > 0:
            y0 = self.calculate_consistent_initial_conditions(
                rhs,
                algebraic,
                model.concatenated_initial_conditions[:, 0],
                jacobian_alg,
            )
        else:
            # can use DAE solver to solve ODE model
            y0 = model.concatenated_initial_conditions[:, 0]

        # Create functions to evaluate residuals
        def residuals(t, y, ydot):
            pybamm.logger.debug(
                "Evaluating residuals for {} at t={}".format(model.name, t)
            )
            y = y[:, np.newaxis]
            rhs_eval, known_evals = concatenated_rhs.evaluate(t, y, known_evals={})
            # reuse known_evals
            alg_eval = concatenated_algebraic.evaluate(t, y, known_evals=known_evals)[0]
            # turn into 1D arrays
            rhs_eval = rhs_eval[:, 0]
            alg_eval = alg_eval[:, 0]
            return (
                np.concatenate((rhs_eval, alg_eval)) - model.mass_matrix.entries @ ydot
            )

        # Create event-dependent function to evaluate events
        def event_fun(event):
            def eval_event(t, y):
                return event.evaluate(t, y)

            return eval_event

        event_funs = [event_fun(event) for event in events.values()]

        # Add the solver attributes
        self.y0 = y0
        self.rhs = rhs
        self.algebraic = algebraic
        self.residuals = residuals
        self.events = events
        self.event_funs = event_funs
        self.jacobian = jacobian

    def set_up_casadi(self, model):
        """Convert model to casadi format and use their inbuilt functionalities.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate. Must have attributes rhs and
            initial_conditions
        """
        # Convert model attributes to casadi
        t_casadi = casadi.SX.sym("t")
        y0 = model.concatenated_initial_conditions
        y_diff = casadi.SX.sym("y_diff", len(model.concatenated_rhs.evaluate(0, y0)))
        y_alg = casadi.SX.sym(
            "y_alg", len(model.concatenated_algebraic.evaluate(0, y0))
        )
        y_casadi = casadi.vertcat(y_diff, y_alg)
        pybamm.logger.info("Converting RHS to CasADi")
        concatenated_rhs = model.concatenated_rhs.to_casadi(t_casadi, y_casadi)
        pybamm.logger.info("Converting algebraic to CasADi")
        concatenated_algebraic = model.concatenated_algebraic.to_casadi(
            t_casadi, y_casadi
        )
        all_states = casadi.vertcat(concatenated_rhs, concatenated_algebraic)
        pybamm.logger.info("Converting events to CasADi")
        casadi_events = {
            name: event.to_casadi(t_casadi, y_casadi)
            for name, event in model.events.items()
        }

        # Create functions to evaluate rhs and algebraic
        concatenated_rhs_fn = casadi.Function(
            "rhs", [t_casadi, y_casadi], [concatenated_rhs]
        )
        concatenated_algebraic_fn = casadi.Function(
            "algebraic", [t_casadi, y_casadi], [concatenated_algebraic]
        )
        all_states_fn = casadi.Function("all", [t_casadi, y_casadi], [all_states])

        if model.use_jacobian:

            pybamm.logger.info("Calculating jacobian")
            casadi_jac = casadi.jacobian(all_states, y_casadi)
            casadi_jac_fn = casadi.Function(
                "jacobian", [t_casadi, y_casadi], [casadi_jac]
            )
            casadi_jac_alg = casadi.jacobian(concatenated_algebraic, y_casadi)
            casadi_jac_alg_fn = casadi.Function(
                "jacobian", [t_casadi, y_casadi], [casadi_jac_alg]
            )

            def jacobian(t, y):
                return casadi_jac_fn(t, y)

            def jacobian_alg(t, y):
                return casadi_jac_alg_fn(t, y)

        else:
            jacobian = None
            jacobian_alg = None

        # Calculate consistent initial conditions for the algebraic equations
        def rhs(t, y):
            return concatenated_rhs_fn(t, y).full()[:, 0]

        def algebraic(t, y):
            return concatenated_algebraic_fn(t, y).full()[:, 0]

        if len(model.algebraic) > 0:
            y0 = self.calculate_consistent_initial_conditions(
                rhs,
                algebraic,
                model.concatenated_initial_conditions[:, 0],
                jacobian_alg,
            )
        else:
            # can use DAE solver to solve ODE model
            y0 = model.concatenated_initial_conditions[:, 0]

        # Create functions to evaluate residuals

        def residuals(t, y, ydot):
            pybamm.logger.debug(
                "Evaluating residuals for {} at t={}".format(model.name, t)
            )
            states_eval = all_states_fn(t, y).full()[:, 0]
            return states_eval - model.mass_matrix.entries @ ydot

        # Create event-dependent function to evaluate events
        def event_fun(event):
            casadi_event_fn = casadi.Function("event", [t_casadi, y_casadi], [event])

            def eval_event(t, y):
                return casadi_event_fn(t, y)

            return eval_event

        event_funs = [event_fun(event) for event in casadi_events.values()]

        # Add the solver attributes
        # Note: these are the (possibly) converted to python version rhs, algebraic
        # etc. The expression tree versions of these are attributes of the model
        self.y0 = y0
        self.rhs = rhs
        self.algebraic = algebraic
        self.residuals = residuals
        self.events = model.events
        self.event_funs = event_funs
        self.jacobian = jacobian

        # Create CasADi problem for the CasADi solver
        self.casadi_problem = {
            "x": y_diff,
            "z": y_alg,
            "ode": concatenated_rhs,
            "alg": concatenated_algebraic,
        }

    def calculate_consistent_initial_conditions(
        self, rhs, algebraic, y0_guess, jac=None
    ):
        """
        Calculate consistent initial conditions for the algebraic equations through
        root-finding

        Parameters
        ----------
        rhs : method
            Function that takes in t and y and returns the value of the differential
            equations
        algebraic : method
            Function that takes in t and y and returns the value of the algebraic
            equations
        y0_guess : array-like
            Array of the user's guess for the initial conditions, used to initialise
            the root finding algorithm
        jac : method
            Function that takes in t and y and returns the value of the jacobian for the
            algebraic equations

        Returns
        -------
        y0_consistent : array-like, same shape as y0_guess
            Initial conditions that are consistent with the algebraic equations (roots
            of the algebraic equations)
        """
        pybamm.logger.info("Start calculating consistent initial conditions")

        # Split y0_guess into differential and algebraic
        len_rhs = rhs(0, y0_guess).shape[0]
        y0_diff, y0_alg_guess = np.split(y0_guess, [len_rhs])

        def root_fun(y0_alg):
            "Evaluates algebraic using y0_diff (fixed) and y0_alg (changed by algo)"
            y0 = np.concatenate([y0_diff, y0_alg])
            out = algebraic(0, y0)
            pybamm.logger.debug(
                "Evaluating algebraic equations at t=0, L2-norm is {}".format(
                    np.linalg.norm(out)
                )
            )
            return out

        if jac:
            if issparse(jac(0, y0_guess)):

                def jac_fn(y0_alg):
                    """
                    Evaluates jacobian using y0_diff (fixed) and y0_alg (varying)
                    """
                    y0 = np.concatenate([y0_diff, y0_alg])
                    return jac(0, y0)[:, len_rhs:].toarray()

            else:

                def jac_fn(y0_alg):
                    """
                    Evaluates jacobian using y0_diff (fixed) and y0_alg (varying)
                    """
                    y0 = np.concatenate([y0_diff, y0_alg])
                    return jac(0, y0)[:, len_rhs:]

        else:
            jac_fn = None
        # Find the values of y0_alg that are roots of the algebraic equations
        sol = optimize.root(
            root_fun,
            y0_alg_guess,
            jac=jac_fn,
            method=self.root_method,
            tol=self.root_tol,
        )
        # Return full set of consistent initial conditions (y0_diff unchanged)
        y0_consistent = np.concatenate([y0_diff, sol.x])

        if sol.success and np.all(sol.fun < self.root_tol * len(sol.x)):
            pybamm.logger.info("Finish calculating consistent initial conditions")
            return y0_consistent
        elif not sol.success:
            raise pybamm.SolverError(
                "Could not find consistent initial conditions: {}".format(sol.message)
            )
        else:
            raise pybamm.SolverError(
                """
                Could not find consistent initial conditions: solver terminated
                successfully, but maximum solution error ({}) above tolerance ({})
                """.format(
                    np.max(sol.fun), self.root_tol * len(sol.x)
                )
            )

    def integrate(
        self, residuals, y0, t_eval, events=None, mass_matrix=None, jacobian=None
    ):
        """
        Solve a DAE model defined by residuals with initial conditions y0.

        Parameters
        ----------
        residuals : method
            A function that takes in t, y and ydot and returns the residuals of the
            equations
        y0 : numeric type
            The initial conditions
        t_eval : numeric type
            The times at which to compute the solution
        events : method, optional
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like, optional
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method, optional
            A function that takes in t, y and ydot and returns the Jacobian
        """
        raise NotImplementedError
