#
# Calculate the Jacobian of a symbol
#
import pybamm


class Jacobian(object):
    def __init__(self, known_jacs=None):
        self._known_jacs = known_jacs or {}

    def jac(self, symbol, variable):
        """
        This function recurses down the tree, computing the Jacobian using
        the Jacobians defined in classes derived from pybamm.Symbol. E.g. the
        Jacobian of a 'pybamm.Multiplication' is computed via the product rule.
        If the Jacobian of a symbol has already been calculated, the stored value
        is returned.
        Note: The Jacobian is the derivative of a symbol with respect to a (slice of)
        a State Vector.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to calculate the Jacobian of
        variable : :class:`pybamm.Symbol`
            The variable with respect to which to differentiate

        Returns
        -------
        :class:`pybamm.Symbol`
            Symbol representing the Jacobian
        """

        try:
            return self._known_jacs[symbol.id]
        except KeyError:
            jac = self._jac(symbol, variable)
            self._known_jacs[symbol.id] = jac
            return jac

    def _jac(self, symbol, variable):
        """ See :meth:`Jacobian.jac()`. """

        if isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            # process children
            left_jac = self.jac(left, variable)
            right_jac = self.jac(right, variable)
            # _binary_jac defined in derived classes for specific rules
            jac = symbol._binary_jac(left_jac, right_jac)

        elif isinstance(symbol, pybamm.UnaryOperator):
            child_jac = self.jac(symbol.child, variable)
            # _unary_jac defined in derived classes for specific rules
            jac = symbol._unary_jac(child_jac)

        elif isinstance(symbol, pybamm.Function):
            children_jacs = [None] * len(symbol.children)
            for i, child in enumerate(symbol.children):
                children_jacs[i] = self.jac(child, variable)
            # _function_jac defined in function class
            jac = symbol._function_jac(children_jacs)

        elif isinstance(symbol, pybamm.Concatenation):
            children_jacs = [child.jac(variable) for child in symbol.cached_children]
            jac = symbol._concatenation_jac(children_jacs)

        else:
            try:
                jac = symbol._jac(variable)
            except NotImplementedError:
                raise NotImplementedError(
                    "Cannot calculate Jacobian of symbol of type '{}'".format(
                        type(symbol)
                    )
                )

        # jacobian removes the domain(s)
        jac.clear_domains()
        return jac
