#
# Class for SEI decomposition in Li-ion batteries
#
import pybamm
from scipy import constants


class SeiDecomposition(pybamm.BaseSubModel):
    """Base class for graphite anode decomposition in Li-ion batteries.

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
        z = pybamm.Variable("Relative SEI thickness", domain="current collector")

        variables = {"Relative SEI thickness": z}
        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        k_b = pybamm.Scalar(constants.k)
        T_av = variables["X-averaged negative electrode temperature"]
        T_av_dimensional = param.Delta_T * T_av + param.T_ref
        c_s_n_surf = variables["X-averaged negative particle surface concentration"]
        z = variables["Relative SEI thickness"]
        x_an = c_s_n_surf

        r_an_dimensional = (
            -param.A_an
            * x_an
            * pybamm.exp(-param.E_an / (k_b * T_av_dimensional))
            * pybamm.exp(-z / param.z_0)
        )  # units 1/s

        Q_scale = param.i_typ * param.potential_scale / param.L_x
        Q_exo_an = -param.rho_n * param.h_an * r_an_dimensional / Q_scale

        variables = {
            "Anode decomposition reaction rate [s-1]": r_an_dimensional,
            "Anode decomposition reaction rate": r_an_dimensional * param.timescale,
            "Anode decomposition heating": Q_exo_an,
            "Anode decomposition heating [W.m-3]": Q_exo_an * Q_scale,
        }

        return variables

    def set_rhs(self, variables):
        decomp_rate = variables["Anode decomposition reaction rate"]
        z = variables["Relative SEI thickness"]

        self.rhs = {z: -decomp_rate}

    def set_initial_conditions(self, variables):
        z = variables["Relative SEI thickness"]
        self.initial_conditions = {z: self.param.z_0}
