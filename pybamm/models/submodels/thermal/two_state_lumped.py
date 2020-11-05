#
# Class for lumped thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class TwoStateLumped(BaseThermal):
    """Class for a two-state (core and surface) lumped thermal submodel

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    cc_dimension: int, optional
        The dimension of the current collectors. Can be 0 (default), 1 or 2.

    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param, cc_dimension=0):
        super().__init__(param, cc_dimension)


    def get_fundamental_variables(self):
        param = self.param

        T_vol_av = pybamm.standard_variables.T_vol_av
        T_x_av = pybamm.PrimaryBroadcast(T_vol_av, ["current collector"])

        T_cn = T_x_av
        T_n = pybamm.PrimaryBroadcast(T_x_av, "negative electrode")
        T_s = pybamm.PrimaryBroadcast(T_x_av, "separator")
        T_p = pybamm.PrimaryBroadcast(T_x_av, "positive electrode")
        T_cp = T_x_av

        T_surface = pybamm.Variable("Surface cell temperature", domain = "current collector")
        T_surface_dim = param.Delta_T * T_surface + param.T_ref

        variables = self._get_standard_fundamental_variables(
            T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
        )
        variables.update({
            "Surface cell temperature": T_surface,
            "Surface cell temperature [K]": T_surface_dim})
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        Q_vol_av = variables["Volume-averaged total heating"]
        T_amb = variables["Ambient temperature"]
        T_surface = variables["Surface cell temperature"] 
        # Account for surface area to volume ratio in cooling coefficient
        # Note: assumes pouch cell geometry. The factor 1/delta^2 comes from
        # the choice of non-dimensionalisation.
        # TODO: allow for arbitrary surface area to volume ratio in order to model
        # different cell geometries (see #718)
        cell_volume = self.param.l * self.param.l_y * self.param.l_z

        yz_cell_surface_area = self.param.l_y * self.param.l_z
        yz_surface_cooling_coefficient = (
            -(self.param.h_cn + self.param.h_cp)
            * yz_cell_surface_area
            / cell_volume
            / (self.param.delta ** 2)
        )

        negative_tab_area = self.param.l_tab_n * self.param.l_cn
        negative_tab_cooling_coefficient = (
            -self.param.h_tab_n * negative_tab_area / cell_volume / self.param.delta
        )

        positive_tab_area = self.param.l_tab_p * self.param.l_cp
        positive_tab_cooling_coefficient = (
            -self.param.h_tab_p * positive_tab_area / cell_volume / self.param.delta
        )

        edge_area = (
            2 * self.param.l_y * self.param.l
            + 2 * self.param.l_z * self.param.l
            - negative_tab_area
            - positive_tab_area
        )
        edge_cooling_coefficient = (
            -self.param.h_edge * edge_area / cell_volume / self.param.delta
        )

        total_cooling_coefficient = (
            yz_surface_cooling_coefficient
            + negative_tab_cooling_coefficient
            + positive_tab_cooling_coefficient
            + edge_cooling_coefficient
        )

        lambda_k = pybamm.x_average(self.param.lambda_k * yz_cell_surface_area / cell_volume / self.param.delta)
        
        self.rhs = {
            T_vol_av: (
                self.param.B * Q_vol_av +  lambda_k * (T_surface - T_vol_av)
            )
            / (self.param.C_th * self.param.rho),
            T_surface: (
                total_cooling_coefficient * (T_surface - T_amb) + lambda_k * (T_vol_av - T_surface)
            )
            / (self.param.C_th * self.param.rho)
        }

    def set_initial_conditions(self, variables):
        T_vol_av = variables["Volume-averaged cell temperature"]
        T_surface = variables["Surface cell temperature"]
        self.initial_conditions = {
            T_vol_av: self.param.T_init, 
            T_surface: self.param.T_init}
