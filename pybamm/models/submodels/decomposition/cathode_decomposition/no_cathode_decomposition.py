#
# Class for cathode decomposition in Li-ion batteries 
#
import pybamm
from scipy import constants 

class NoCathodeDecomposition(pybamm.BaseSubModel):
    """Base class for cathode decomposition in Li-ion batteries.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, reactions=None):
        super().__init__(param, reactions=reactions)

    def get_fundamental_variables(self):
        
        variables = {
            "Degree of conversion of cathode decomposition": pybamm.Scalar(0),
            "Cathode decomposition reaction rate [s-1]": pybamm.Scalar(0),
            "Cathode decomposition reaction rate": pybamm.Scalar(0),
            "Cathode decomposition heating": pybamm.Scalar(0),
        }
        return variables