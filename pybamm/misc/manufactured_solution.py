#
# Manufactured solution class
#
import pybamm

import copy
import autograd.numpy as np


class ManufacturedSolution(object):
    """
    Creates a manufactured solution for a particular model, for testing convergence of
    the solver.

    Notes
    -----

    The Method of Manufactured Solutions is a code-checking technique that consists in
    choosing a particular expression, seeing what the source terms, initial conditions
    and boundary conditions would need to be for that expression to be a solution of the
    equations, and then solving the equations with those source terms, initial
    conditions and boundary conditions and checking that the solution converges to the
    chosen expression in the right way.

    This is useful since we cannot in general find an explicit analytical solution to
    compare our model to.

    For example, say we want to solve the equation

    .. math::
        \\frac{\\partial c}{\\partial t}
        = \\frac{\\partial^2 c}{\\partial x^2}

    with Dirichlet boundary conditions. if we choose the manufactured solution,

    .. math::
        c_M = \\exp(t)\\sin(2 \\pi x)

    then the system that we want to solve is

    .. math::
        \\frac{\\partial c}{\\partial t}
        = \\frac{\\partial^2 c}{\\partial x^2}
        + (1 + 4 \\pi^2)\\exp(t)\\sin(2 \\pi x)

        c(x,0) = \\sin(2 \\pi x)

        c(0,t) = c(1,t) = 0

    We can then solve this system and check that the solution converges to :math:`c_M`
    in the expected way as we refine the grid size.

    For more details, see for example `Sandia online notes
    <https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2000/001444.pdf>`_.
    """

    def process_model(self, model, manufactured_variables=None):
        """
        Create manufactured solution from model:

        * Add source term to equations
        * Set boundary conditions
        * Set initial condition

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model to create a manufactured solution for
        manufactured_variables : dict, optional
            Dictionary '{variable id: expression}' of manufactured variables. Default is
            None, in which case random manufactured solutions are created using
            :meth:`create_manufactured_variable`

        """
        # Read and create manufactured variables
        model_variables = model.initial_conditions.keys()
        if manufactured_variables is None:
            # Create dictionary of manufactured variables
            manufactured_variables = {
                var.id: self.create_manufactured_variable(var.domain)
                for var in model_variables
            }
        self.manufactured_variables = manufactured_variables
        self.set_manufactured_variable_strings(model.variables)

        # Add appropriate source terms to the equations
        for var, eqn in model.rhs.items():
            # Calculate source term
            source_term = -self.process_symbol(eqn)
            # Calculate additional source term for the differential part
            source_term += self.manufactured_variables[var.id].diff(pybamm.t)
            # Add source term to equation
            model.rhs[var] += source_term
        for var, eqn in model.algebraic.items():
            # Calculate source term
            source_term = -self.process_symbol(eqn)
            # Add source term to equation
            model.algebraic[var] += source_term

        # Set initial conditions using manufactured variables
        # (for the algebraic equations, these should already be consistent with eqns)
        model.initial_conditions = {
            var: manufactured_variables[var.id] for var in model_variables
        }

        # Set boundary conditions using manufactured variables
        for expr in model.boundary_conditions:
            expr_proc = self.process_symbol(expr)
            # The boundary condition is the processed expression evaluated on that
            # boundary
            model.boundary_conditions[expr] = {
                side: pybamm.BoundaryValue(expr_proc, side)
                for side in ["left", "right"]
            }

    def set_manufactured_variable_strings(self, model_variables):
        """
        Create dictionary pointing to manufactured variables, using model.variables
        (not infallible). This is used in testing to compare manufactured variables
        to the solution of the model.


        Parameters
        ----------
        model_variables : dict
            Dictionary '{string: expression}' of model variables.

        Raises
        ------
        pybamm.ModelError
            If no object in ``model_variables.values`` (or its child if it is a
            :class:`pybamm.Broadcast`) has the same id as any key of
            ``self.manufactured_variables``.
        """
        self.manufactured_variable_strings = {}
        for var_string, var_expr in model_variables.items():
            if var_expr.id in self.manufactured_variables:
                self.manufactured_variable_strings[
                    var_string
                ] = self.manufactured_variables[var_expr.id]
            # also check for children of a `Broadcast`
            elif (
                isinstance(var_expr, pybamm.Broadcast)
                and var_expr.children[0].id in self.manufactured_variables
            ):
                self.manufactured_variable_strings[var_string] = pybamm.Broadcast(
                    self.manufactured_variables[var_expr.children[0].id],
                    var_expr.domain,
                )
        if self.manufactured_variable_strings == {}:
            raise ValueError(
                """No manufactured variables can be found in model variables"""
            )

    def create_manufactured_variable(self, domain):
        """
        Return a randomly chosen manufactured variable on the appropriate domain from a
        list of possible options

        Parameters
        ----------
        domain : iterable of str
            The domain on which to create the manufactured variable

        Returns
        -------
        :class:`pybamm.Symbol`
            A manufactured variable (function of t and x and/or r) on the appropriate
            domain
        """
        t = pybamm.t
        if domain == []:
            x = pybamm.Scalar(np.random.rand())
            r = pybamm.Scalar(np.random.rand())
        elif domain in [["negative particle"], ["positive particle"]]:
            x = pybamm.Scalar(np.random.rand())
            r = pybamm.SpatialVariable("r", domain=domain)
        else:
            x = pybamm.SpatialVariable("x", domain=domain)
            r = pybamm.Scalar(np.random.rand())

        # Construct random forms of variables
        # random types
        # random coefficients
        # either x is a scalar or r is a scalar, so there is no domain clash
        a, b, c = np.random.rand(3)
        options = [
            a * t + b * r + c * x,
            a * t * b * pybamm.Function(np.cos, c * r * x),
            a * pybamm.Function(np.exp, b * t) * (c + r * x) ** 2,
        ]
        return options[np.random.randint(len(options))]

    def process_symbol(self, symbol):
        """Walk through the symbol and replace any Variable with a manufactured variable
        and any Gradient or Divergence with the explicit derivative

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol or Expression tree to process

        Returns
        -------
        symbol : :class:`pybamm.Symbol`
            Symbol with Variable instances replaced by manufactured variables

        """
        if isinstance(symbol, pybamm.Variable):
            return self.manufactured_variables[symbol.id]

        elif isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            new_left = self.process_symbol(left)
            new_right = self.process_symbol(right)
            # make new symbol, ensure domain remains the same
            new_symbol = symbol.__class__(new_left, new_right)
            new_symbol.domain = symbol.domain
            return new_symbol

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.children[0])
            if isinstance(symbol, pybamm.Gradient):
                new_symbol = self.gradient(new_child)
            elif isinstance(symbol, pybamm.Divergence):
                new_symbol = self.divergence(new_child)
            elif isinstance(symbol, pybamm.NumpyBroadcast):
                new_symbol = pybamm.NumpyBroadcast(
                    new_child, symbol.domain, symbol.mesh
                )
            elif isinstance(symbol, pybamm.Broadcast):
                new_symbol = pybamm.Broadcast(new_child, symbol.domain)
            elif isinstance(symbol, pybamm.Function):
                new_symbol = pybamm.Function(symbol.func, new_child)
            elif isinstance(symbol, pybamm.Integral):
                new_symbol = pybamm.Integral(new_child, symbol.integration_variable)
            elif isinstance(symbol, pybamm.BoundaryValue):
                new_symbol = pybamm.BoundaryValue(new_child, symbol.side)
            else:
                new_symbol = symbol.__class__(new_child)
            # ensure domain remains the same
            new_symbol.domain = symbol.domain
            return new_symbol

        # Concatenations
        elif isinstance(symbol, pybamm.Concatenation):
            new_children = []
            for child in symbol.children:
                new_child = self.process_symbol(child)
                new_children.append(new_child)
            if isinstance(symbol, pybamm.DomainConcatenation):
                return pybamm.DomainConcatenation(new_children, symbol.mesh)
            else:
                # Concatenation or NumpyConcatenation
                return symbol.__class__(*new_children)

        else:
            new_symbol = copy.deepcopy(symbol)
            new_symbol.parent = None
            return new_symbol

    def gradient(self, symbol):
        """
        Symbolically calculate the gradient

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol whose gradient to calculate

        Returns
        -------
        :class:`pybamm.Symbol`
            The symbolic gradient of the symbol with respect to x (if symbol domain is
            macroscale) or r (if symbol domain is microscale).
        """
        domain = symbol.domain
        if domain == []:
            return pybamm.Scalar(0)
        elif domain in [["negative particle"], ["positive particle"]]:
            r = pybamm.SpatialVariable("r", domain=domain)
            return symbol.diff(r)
        else:
            x = pybamm.SpatialVariable("x", domain=domain)
            return symbol.diff(x)

    def divergence(self, symbol):
        """
        Symbolically calculate the divergence

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol whose gradient to calculate

        Returns
        -------
        :class:`pybamm.Symbol`
            The symbolic divergence of the symbol with respect to x (if symbol domain is
            macroscale) or r (if symbol domain is microscale).
        """
        domain = symbol.domain
        if domain == []:
            return pybamm.Scalar(0)
        elif domain in [["negative particle"], ["positive particle"]]:
            r = pybamm.SpatialVariable("r", domain=domain)
            return 1 / (r ** 2) * (r ** 2 * symbol).diff(r)
        else:
            x = pybamm.SpatialVariable("x", domain=domain)
            return symbol.diff(x)
