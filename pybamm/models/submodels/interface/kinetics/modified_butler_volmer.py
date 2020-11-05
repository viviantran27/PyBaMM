#
# Modified bulter volmerclass
#

import pybamm
from scipy import constants
from .base_kinetics import BaseKinetics


class ModifiedButlerVolmer(BaseKinetics):
    """
    Base submodel which implements the forward Butler-Volmer equation:

    .. math::
        j = 2 * j_0(c) * \\sinh( (ne / (2 * (1 + \\Theta T)) * \\eta_r(c))/
            (1 - j_0/j_lim exp(-0.5 * F* eta_r(c) / R * T))

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. In this case "sei film
        resistance" is the important option. See :class:`pybamm.BaseBatteryModel`

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction, options=None):
        super().__init__(param, domain, reaction, options)

    def _get_kinetics(self, j0, ne, eta_r, T, variables):
        F = pybamm.Scalar(self.param.F)
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        c_e_0 = variables["X-averaged electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Postive electrolyte concentration [mol.m-3]"]
        c_e_surf_p = pybamm.surf(c_e_p)
        delta_c_e = c_e_0 - c_e_surf_p
        T_av = variables["X-averaged cell temperature [K]"]
        D_e = self.param.D_e_dimensional(c_e_surf_p,T_av)
        delta_e = pybamm.sqrt(constants.pi * D_e * pybamm.t)
        c_e_lim = j0 * delta_e / F / D_e 
        return 2 * j0 * pybamm.sinh(prefactor * eta_r) / (1 + c_e_lim / delta_c_e * pybamm.exp(prefactor* eta_r)) 

    def _get_dj_dc(self, variables):
        "See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_dc`"
        c_e, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return (2 * j0.diff(c_e) * pybamm.sinh(prefactor * eta_r)) - (
            2 * j0 * prefactor * ocp.diff(c_e) * pybamm.cosh(prefactor * eta_r)
        )

    def _get_dj_ddeltaphi(self, variables):
        "See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_ddeltaphi`"
        _, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        prefactor = ne / (2 * (1 + self.param.Theta * T))
        return 2 * j0 * prefactor * pybamm.cosh(prefactor * eta_r)
