#
# Manufactured solution class
#
import pybamm

import copy
import numpy as np


class ManufacturedSolution(object):
    """

    """

    def process_model(self, model):
        # Read and create manufactured variables
        man_vars = {
            var.id: self.create_manufactured_variable(var.domain)
            for var in model.initial_conditions.keys()
        }

        # Add appropriate source terms to the equations
        for var, eqn in model.rhs.items():
            # Calculate source term
            source_term = -self.process_symbol(eqn, man_vars)
            # Calculate additional source term for the differential part
            source_term += var_dt.diff(pybamm.t)
            # Add source term to equation
            model.rhs[var] += source_term
        for var, eqn in model.algebraic.items():
            # Calculate source term
            source_term = -self.process_symbol(eqn, man_vars)
            # Add source term to equation
            model.algebraic[var] += source_term

        # Set initial conditions using manufactured variables
        # (for the algebraic equations, these should already be consistent with eqns)
        model.initial_conditions = {var: man_vars[var] for var in man_vars}

        # Set boundary conditions using manufactured variables
        for expr in model.boundary_conditions:
            expr_proc = self.process_symbol(expr, man_vars)
            # The boundary condition is the processed expression evaluated on that
            # boundary
            model.boundary_conditions[expr] = {
                side: pybamm.BoundaryValue(expr_proc, side)
                for side in ["left", "right"]
            }

    def create_manufactured_variable(self, domain):
        t = pybamm.t
        if domain == []:
            x = pybamm.Scalar(1)
            r = pybamm.Scalar(1)
        elif domain in [["negative particle"], ["positive particle"]]:
            x = pybamm.Scalar(1)
            r = pybamm.SpatialVariable("r", domain=domain)
        else:
            x = pybamm.SpatialVariable("x", domain=domain)
            r = pybamm.Scalar(1)

        # Construct random forms of variables
        # random types
        # random coefficients
        options = [t * pybamm.Function(np.cos, r)]
        return options[0]

    def process_symbol(self, symbol, variables):
        """Walk through the symbol and replace any Variable with a manufactured variable
        and any Gradient or Divergence with the explicit derivative

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            Symbol or Expression tree to process
        variables : list
            List of manufactured variables

        Returns
        -------
        symbol : :class:`pybamm.Symbol` (or subclass) instance
            Symbol with Variable instances replaced by manufactured variables

        """
        if isinstance(symbol, pybamm.Variable):
            return variables[symbol.id]

        elif isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            new_left = self.process_symbol(left, variables)
            new_right = self.process_symbol(right, variables)
            # make new symbol, ensure domain remains the same
            new_symbol = symbol.__class__(new_left, new_right)
            new_symbol.domain = symbol.domain
            return new_symbol

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.process_symbol(symbol.children[0], variables)
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
                new_child = self.process_symbol(child, variables)
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
        domain = symbol.domain
        if domain == []:
            return pybamm.Scalar(0)
        elif domain in [["negative particle"], ["positive particle"]]:
            r = pybamm.SpatialVariable("r", domain=domain)
            return 1 / (r ** 2) * (r ** 2 * symbol).diff(r)
        else:
            x = pybamm.SpatialVariable("x", domain=domain)
            return symbol.diff(x)
