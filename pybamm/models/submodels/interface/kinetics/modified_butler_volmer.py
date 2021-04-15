#
# Bulter volmer class
#

import pybamm
from .base_kinetics import BaseKinetics


class ModifiedButlerVolmer(BaseKinetics):
    """
    Base submodel which implements the forward Butler-Volmer equation:

    .. math::
        j = 2 * j_0(c) * \\sinh( (ne / (2 * (1 + \\Theta T)) * \\eta_r(c))

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`

    **Extends:** :class:`pybamm.interface.kinetics.BaseKinetics`
    """

    def __init__(self, param, domain, reaction, options):
        super().__init__(param, domain, reaction, options)

    # def _get_kinetics(self, j0, ne, eta_r, T, variables):
    #     prefactor = ne / (2 * (1 + self.param.Theta * T))
    #     return 2 * j0 * pybamm.sinh(prefactor * eta_r)

    def _get_kinetics(self, j0, ne, eta_r, T, variables):
        k1 = ne / (2 * (1 + self.param.Theta * T))
        # F = pybamm.Scalar(self.param.F)
        # T_av = variables["X-averaged cell temperature [K]"]
        # D_e = self.param.D_e_dimensional(c_e_surf_p,T_av)
        # delta_e = pybamm.sqrt(constants.pi * D_e * pybamm.t)
        
        c_e_lim = 1     #[mol.m-3] constant from paper
        c_s_lim = 1e-4  #[mol.m-3] constant from paper
        c_e_0 = variables["X-averaged electrolyte concentration [mol.m-3]"]

        if self.domain == "Positive": #Eqn 12
            c_e_p = variables["Postive electrolyte concentration [mol.m-3]"]
            c_e_surf_p = pybamm.surf(c_e_p)
            delta_c_e = c_e_0 - c_e_surf_p

            c_s_p = variables["Postive particle concentration [mol.m-3]"]
            c_s_surf_p = pybamm.surf(c_s_p)
            delta_c_s = c_s_p - c_s_surf_p

            k2 = c_e_lim/delta_c_e +  c_s_lim/delta_c_s

        else: #Eqn 13
            c_s_n = variables["Negative particle concentration [mol.m-3]"]
            c_s_surf_n = pybamm.surf(c_s_n)
            delta_c_s = c_s_n - c_s_surf_n

            k2 = c_s_lim/delta_c_s

        return 2 * j0 * pybamm.sinh(k1 * eta_r)/ (1 + k2 * pybamm.exp(-k1 * eta_r))

    # def _get_dj_dc(self, variables):
    #     """ See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_dc` """
    #     c_e, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
    #         variables
    #     )
    #     eta_r = delta_phi - ocp
    #     prefactor = ne / (2 * (1 + self.param.Theta * T))
    #     return (2 * j0.diff(c_e) * pybamm.sinh(prefactor * eta_r)) - (
    #         2 * j0 * prefactor * ocp.diff(c_e) * pybamm.cosh(prefactor * eta_r)
    #     )

    # def _get_dj_ddeltaphi(self, variables):
    #     """ See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_ddeltaphi` """
    #     _, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
    #         variables
    #     )
    #     eta_r = delta_phi - ocp
    #     prefactor = ne / (2 * (1 + self.param.Theta * T))
    #     return 2 * j0 * prefactor * pybamm.cosh(prefactor * eta_r)

    def _get_dj_ddeltaphi(self, variables):
        """ See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_ddeltaphi` """
        _, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        k1 = ne / (2 * (1 + self.param.Theta * T))

        # calculate k2
        c_e_lim = 1     #[mol.m-3] constant from paper
        c_s_lim = 1e-4  #[mol.m-3] constant from paper
        c_e_0 = variables["X-averaged electrolyte concentration [mol.m-3]"]

        if self.domain == "Positive": #Eqn 12
            c_e_p = variables["Postive electrolyte concentration [mol.m-3]"]
            c_e_surf_p = pybamm.surf(c_e_p)
            delta_c_e = c_e_0 - c_e_surf_p

            c_s_p = variables["Postive particle concentration [mol.m-3]"]
            c_s_surf_p = pybamm.surf(c_s_p)
            delta_c_s = c_s_p - c_s_surf_p

            k2 = c_e_lim/delta_c_e +  c_s_lim/delta_c_s

        else: #Eqn 13
            c_s_n = variables["Negative particle concentration [mol.m-3]"]
            c_s_surf_n = pybamm.surf(c_s_n)
            delta_c_s = c_s_n - c_s_surf_n

            k2 = c_e_lim/delta_c_e +  c_s_lim/delta_c_s

        # quotient rule
        f = pybamm.sinh(k1 * eta_r)
        df_deta = k1 * pybamm.cosh(k1 * eta_r)

        g = 1 + k2 * pybamm.exp(-k1 * eta_r)
        dg_deta = -k1 * k2 * pybamm.exp(-k1 * eta_r) 
        return 2 * j0 * (df_deta * g - f * dg_deta)/ g**2

    def _get_dj_dc(self, variables):
        """ See :meth:`pybamm.interface.kinetics.BaseKinetics._get_dj_dc` """
        c_e, delta_phi, j0, ne, ocp, T = self._get_interface_variables_for_first_order(
            variables
        )
        eta_r = delta_phi - ocp
        k1 = ne / (2 * (1 + self.param.Theta * T))

        # calculate k2
        c_e_lim = 1     #[mol.m-3] constant from paper
        c_s_lim = 1e-4  #[mol.m-3] constant from paper
        c_e_0 = variables["X-averaged electrolyte concentration [mol.m-3]"]

        if self.domain == "Positive": #Eqn 12
            c_e_p = variables["Postive electrolyte concentration [mol.m-3]"]
            c_e_surf_p = pybamm.surf(c_e_p)
            delta_c_e = c_e_0 - c_e_surf_p

            c_s_p = variables["Postive particle concentration [mol.m-3]"]
            c_s_surf_p = pybamm.surf(c_s_p)
            delta_c_s = c_s_p - c_s_surf_p

            k2 = c_e_lim/delta_c_e +  c_s_lim/delta_c_s

        else: #Eqn 13
            c_s_n = variables["Negative particle concentration [mol.m-3]"]
            c_s_surf_n = pybamm.surf(c_s_n)
            delta_c_s = c_s_n - c_s_surf_n

            k2 = c_e_lim/delta_c_e +  c_s_lim/delta_c_s


        # quotient rule
        f = pybamm.sinh(k1 * eta_r)
        df_deta = k1 * pybamm.cosh(k1 * eta_r)

        g = 1 + k2 * pybamm.exp(-k1 * eta_r)
        dg_deta = -k1 * k2 * pybamm.exp(-k1 * eta_r) 

        return (2 * j0.diff(c_e) * pybamm.sinh(k1 * eta_r) / (1 + k2 * pybamm.exp(-k1 * eta_r))
            + ocp.diff(c_e) * 2 * j0 * (df_deta * g - f * dg_deta)/ g**2)
