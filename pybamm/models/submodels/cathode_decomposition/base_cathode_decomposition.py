#
# Class for cathode decomposition in Li-ion batteries 
#
import pybamm
from scipy import constants 

class CathodeDecomposition(pybamm.BaseSubModel):
    """Base class for cathode decomposition in Li-ion batteries.

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
        alpha = pybamm.Variable("Degree of conversion of cathode decomposition", domain="current collector")
        
        variables = {"Degree of conversion of cathode decomposition": alpha}
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        k_b = pybamm.Scalar(constants.k) 
        T_av = variables["X-averaged positive electrode temperature"]
        T_av_dimensional = param.Delta_T * T_av + param.T_ref
        alpha = variables["Degree of conversion of cathode decomposition"]

        r_ca_dimensional = (
            alpha
            * (1-alpha) 
            * param.A_ca 
            * pybamm.exp(-param.E_ca/(k_b*T_av_dimensional))
        ) # units 1/s

        Q_scale = param.i_typ * param.potential_scale / param.L_x
        Q_exo_ca = param.rho_p * param.h_ca * r_ca_dimensional / Q_scale

        variables = {
            "Cathode decomposition reaction rate [s-1]": r_ca_dimensional,
            "Cathode decomposition reaction rate": r_ca_dimensional * param.timescale,
            "Cathode decomposition heating": Q_exo_ca,
            "Cathode decomposition heating [W.m-3]": Q_exo_ca * Q_scale,
        }
        return variables

    def set_rhs(self, variables):
        decomp_rate = variables["Cathode decomposition reaction rate"]
        alpha = variables["Degree of conversion of cathode decomposition"]
        
        self.rhs = {alpha: decomp_rate}

    def set_initial_conditions(self, variables):
        alpha = variables["Degree of conversion of cathode decomposition"]
        self.initial_conditions = {alpha: self.param.alpha_0}