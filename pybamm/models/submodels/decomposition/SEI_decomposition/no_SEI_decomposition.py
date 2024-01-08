#
# Class for when there is no anode decomposition
#
import pybamm
from scipy import constants


class NoSeiDecomposition(pybamm.BaseSubModel):
    """Base class for no graphite anode decomposition in Li-ion batteries.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, reactions=None):
        super().__init__(param)

    def get_fundamental_variables(self):
        default = pybamm.Scalar(0)
        x_sei = self.param.therm.x_sei_0

        variables = {
            "Fraction of Li in SEI": x_sei,
            "Fraction of Li in SEI in the core section": x_sei,
            "Fraction of Li in SEI in the middle section": x_sei,
            "Fraction of Li in SEI in the outer section": x_sei,

            "SEI decomposition reaction rate [s-1]": default,
            "SEI decomposition reaction rate": default,
            "SEI decomposition heating": default,
            "SEI decomposition heating [W.m-3]": default,

            "Core section SEI decomposition reaction rate [s-1]": default,
            "Core section SEI decomposition reaction rate": default,
            "Core section SEI decomposition heating": default,
            "Core section SEI decomposition heating [W.m-3]": default,

            "Middle section SEI decomposition reaction rate [s-1]": default,
            "Middle section SEI decomposition reaction rate": default,
            "Middle section SEI decomposition heating": default,
            "Middle section SEI decomposition heating [W.m-3]": default,

            "Outer section SEI decomposition reaction rate [s-1]": default,
            "Outer section SEI decomposition reaction rate": default,
            "Outer section SEI decomposition heating": default,
            "Outer section SEI decomposition heating [W.m-3]": default,
        }

        return variables
