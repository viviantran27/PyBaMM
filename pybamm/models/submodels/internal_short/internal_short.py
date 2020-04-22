#
# Class for internal short circuit in Li-ion batteries 
#
import pybamm
from scipy import constants 

class InternalShort(pybamm.BaseSubModel):
    """Base class for interal short circuit.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    reactions : dict, optional
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        I_short = pybamm.Variable("Internal short circuit current [A]")
        
        variables = {"Internal short circuit current [A]": I_short}
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        I_short = variables["Internal short circuit current [A]"]
        r_an_short_dimensional = -param.c_n_init(0)*I_short/param.Q)
        variables = {
            "Cathode decomposition reaction rate [s-1]": r_ca_dimensional,
            "Cathode decomposition reaction rate": r_ca_dimensional * param.timescale,
        }
        return variables

    def set_rhs(self, variables):
        decomp_rate = variables["Cathode decomposition reaction rate"]
        alpha = variables["Degree of conversion of cathode decomposition"]
        
        self.rhs = {alpha: decomp_rate}

    def set_initial_conditions(self, variables):
        alpha = variables["Degree of conversion of cathode decomposition"]
        self.initial_conditions = {alpha: self.param.alpha_0}