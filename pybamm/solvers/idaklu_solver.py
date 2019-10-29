#
# Solver class using sundials with the KLU sparse linear solver
#
import pybamm
import numpy as np
import scipy.sparse as sparse

import importlib

idaklu_spec = importlib.util.find_spec("idaklu")
if idaklu_spec is not None:
    idaklu = importlib.util.module_from_spec(idaklu_spec)
    idaklu_spec.loader.exec_module(idaklu)


def have_idaklu():
    return idaklu_spec is None


class IDAKLUSolver(pybamm.DaeSolver):
    """Solve a discretised model, using sundials with the KLU sparse linear solver.

     Parameters
    ----------
    rtol : float, optional
        The relative tolerance for the solver (default is 1e-6).
    atol : float, optional
        The absolute tolerance for the solver (default is 1e-6).
    root_method : str, optional
        The method to use to find initial conditions (default is "lm")
    root_tol : float, optional
        The tolerance for the initial-condition solver (default is 1e-8).
    max_steps: int, optional
        The maximum number of steps the solver will take before terminating
        (default is 1000).
    """

    def __init__(
        self, rtol=1e-6, atol=1e-6, root_method="lm", root_tol=1e-6, max_steps=1000
    ):

        if idaklu_spec is None:
            raise ImportError("KLU is not installed")

        super().__init__("ida", rtol, atol, root_method, root_tol, max_steps)

    def integrate(self, residuals, y0, t_eval, events, mass_matrix, jacobian):
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
        events : method,
            A function that takes in t and y and returns conditions for the solver to
            stop
        mass_matrix : array_like,
            The (sparse) mass matrix for the chosen spatial method.
        jacobian : method,
            A function that takes in t and y and returns the Jacobian. If
            None, the solver will approximate the Jacobian.
            (see `SUNDIALS docs. <https://computation.llnl.gov/projects/sundials>`).
        """

        if jacobian is None:
            pybamm.SolverError("KLU requires the Jacobian to be provided")

        if events is None:
            pybamm.SolverError("KLU requires events to be provided")

        rtol = self._rtol
        atol = self._atol

        if jacobian:
            jac_y0_t0 = jacobian(t_eval[0], y0)
            if sparse.issparse(jac_y0_t0):

                def jacfn(t, y, cj):
                    j = jacobian(t, y) - cj * mass_matrix
                    return j

            else:

                def jacfn(t, y, cj):
                    jac_eval = jacobian(t, y) - cj * mass_matrix
                    return sparse.csr_matrix(jac_eval)

        class SundialsJacobian:
            def __init__(self):
                self.J = None

                random = np.random.random(size=y0.size)
                J = jacfn(10, random, 20)
                self.nnz = J.nnz  # hoping nnz remains constant...

            def jac_res(self, t, y, cj):
                # must be of form j_res = (dr/dy) - (cj) (dr/dy')
                # cj is just the input parameter
                # see p68 of the ida_guide.pdf for more details
                self.J = jacfn(t, y, cj)

            def get_jac_data(self):
                return self.J.data

            def get_jac_row_vals(self):
                return self.J.indices

            def get_jac_col_ptrs(self):
                return self.J.indptr

        # solver works with ydot0 set to zero
        ydot0 = np.zeros_like(y0)

        jac_class = SundialsJacobian()

        num_of_events = len(events)
        use_jac = 1

        def rootfn(t, y):
            return_root = np.ones((num_of_events,))
            return_root[:] = [event(t, y) for event in events]

            return return_root

        # get ids of rhs and algebraic variables
        rhs_ids = np.ones(self.rhs(0, y0).shape)
        alg_ids = np.zeros(self.algebraic(0, y0).shape)
        ids = np.concatenate((rhs_ids, alg_ids))

        # solve
        sol = idaklu.solve(
            t_eval,
            y0,
            ydot0,
            self.residuals,
            jac_class.jac_res,
            jac_class.get_jac_data,
            jac_class.get_jac_row_vals,
            jac_class.get_jac_col_ptrs,
            jac_class.nnz,
            rootfn,
            num_of_events,
            use_jac,
            ids,
            rtol,
            atol,
        )

        t = sol.t
        number_of_timesteps = t.size
        number_of_states = y0.size
        y_out = sol.y.reshape((number_of_timesteps, number_of_states))

        # return solution, we need to tranpose y to match scipy's interface
        if sol.flag in [0, 2]:
            # 0 = solved for all t_eval
            if sol.flag == 0:
                termination = "final time"
            # 2 = found root(s)
            elif sol.flag == 2:
                termination = "event"
            return pybamm.Solution(
                sol.t, np.transpose(y_out), t[-1], np.transpose(y_out[-1]), termination
            )
        else:
            raise pybamm.SolverError(sol.message)
