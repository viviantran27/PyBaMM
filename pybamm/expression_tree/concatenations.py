#
# Concatenation classes
#
import pybamm

import numpy as np
from scipy.sparse import vstack
import copy


class Concatenation(pybamm.Symbol):
    """A node in the expression tree representing a concatenation of symbols

    **Extends**: :class:`pybamm.Symbol`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    """

    def __init__(self, *children, name=None, check_domain=True):
        if name is None:
            name = "concatenation"
        if check_domain:
            domain = self.get_children_domains(children)
        else:
            domain = []
        super().__init__(name, children, domain=domain)

    def get_children_domains(self, children):
        # combine domains from children
        domain = []
        for child in children:
            child_domain = child.domain
            if set(domain).isdisjoint(child_domain):
                domain += child_domain
            else:
                raise pybamm.DomainError("""domain of children must be disjoint""")
        return domain

    def _concatenation_evaluate(self, children_eval):
        """ Concatenate the evaluated children. """
        raise NotImplementedError

    def evaluate(self, t=None, y=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        children = self.cached_children
        if known_evals is not None:
            if self.id not in known_evals:
                children_eval = [None] * len(children)
                for idx, child in enumerate(children):
                    children_eval[idx], known_evals = child.evaluate(t, y, known_evals)
                known_evals[self.id] = self._concatenation_evaluate(children_eval)
            return known_evals[self.id], known_evals
        else:
            children_eval = [None] * len(children)
            for idx, child in enumerate(children):
                children_eval[idx] = child.evaluate(t, y)
            return self._concatenation_evaluate(children_eval)

    def _concatenation_simplify(self, children):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        new_symbol = self.__class__(*children)
        new_symbol.domain = []
        return new_symbol


class NumpyConcatenation(Concatenation):
    """A node in the expression tree representing a concatenation of equations, when we
    *don't* care about domains.

    Upon evaluation, equations are concatenated using numpy concatenation.

    **Extends**: :class:`Concatenation`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The equations to concatenate

    """

    def __init__(self, *children):
        children = list(children)
        # Turn objects that evaluate to scalars to objects that evaluate to vectors,
        # so that we can concatenate them
        for i, child in enumerate(children):
            if child.evaluates_to_number():
                children[i] = child * pybamm.Vector(np.array([1]))
        super().__init__(*children, name="numpy concatenation", check_domain=False)

    def _concatenation_evaluate(self, children_eval):
        """ See :meth:`Concatenation._concatenation_evaluate()`. """
        if len(children_eval) == 0:
            return np.array([])
        else:
            return np.concatenate([child for child in children_eval])

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        children = self.cached_children
        if len(children) == 0:
            return pybamm.Scalar(0)
        else:
            return SparseStack(*[child.jac(variable) for child in children])


class SparseStack(Concatenation):
    """A node in the expression tree representing a concatenation of sparse
    matrices. As with NumpyConcatenation, we *don't* care about domains.

    **Extends**: :class:`Concatenation`

    Parameters
    ----------
    children : iterable of :class:`Concatenation`
        The equations to concatenate

    """

    def __init__(self, *children):
        children = list(children)
        super().__init__(*children, name="sparse stack", check_domain=False)

    def evaluate(self, t=None, y=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        children = self.cached_children
        if known_evals is not None:
            if self.id not in known_evals:
                children_eval = [None] * len(children)
                for idx, child in enumerate(children):
                    children_eval[idx], known_evals = child.evaluate(t, y, known_evals)
                known_evals[self.id] = self._concatenation_evaluate(children_eval)
            return known_evals[self.id], known_evals
        else:
            children_eval = [None] * len(children)
            for idx, child in enumerate(children):
                children_eval[idx] = child.evaluate(t, y)
            return self._concatenation_evaluate(children_eval)

    def _concatenation_evaluate(self, children_eval):
        """ See :meth:`Concatenation.evaluate()`. """
        return vstack(children_eval)
