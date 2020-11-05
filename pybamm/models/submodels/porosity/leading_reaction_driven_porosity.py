#
# Class for leading-order reaction driven porosity changes
#
import pybamm
from .base_porosity import BaseModel


class LeadingOrder(BaseModel):
    """Leading-order model for reaction-driven porosity changes

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.porosity.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        eps_n_pc = pybamm.standard_variables.eps_n_pc
        eps_s_pc = pybamm.standard_variables.eps_s_pc
        eps_p_pc = pybamm.standard_variables.eps_p_pc

        eps_n = pybamm.PrimaryBroadcast(eps_n_pc, "negative electrode")
        eps_s = pybamm.PrimaryBroadcast(eps_s_pc, "separator")
        eps_p = pybamm.PrimaryBroadcast(eps_p_pc, "positive electrode")

        variables = self._get_standard_porosity_variables(eps_n, eps_s, eps_p)
        return variables

    def get_coupled_variables(self, variables):

        j_n = variables["X-averaged negative electrode interfacial current density"]
        j_p = variables["X-averaged positive electrode interfacial current density"]
        j_sei_n = variables[
            "X-averaged negative electrode sei interfacial current density"
        ]
        beta_sei_n = self.param.beta_sei_n

        deps_n_dt = pybamm.PrimaryBroadcast(
            -self.param.beta_surf_n * j_n + beta_sei_n * j_sei_n, ["negative electrode"]
        )
        deps_s_dt = pybamm.FullBroadcast(
            0, "separator", auxiliary_domains={"secondary": "current collector"}
        )
        deps_p_dt = pybamm.PrimaryBroadcast(
            -self.param.beta_surf_p * j_p, ["positive electrode"]
        )

        variables.update(
            self._get_standard_porosity_change_variables(
                deps_n_dt, deps_s_dt, deps_p_dt
            )
        )

        return variables

    def set_rhs(self, variables):
        self.rhs = {}
        for domain in ["negative electrode", "separator", "positive electrode"]:
            eps_av = variables["X-averaged " + domain + " porosity"]
            deps_dt_av = variables["X-averaged " + domain + " porosity change"]
            self.rhs.update({eps_av: deps_dt_av})

    def set_initial_conditions(self, variables):

        eps_n_av = variables["X-averaged negative electrode porosity"]
        eps_s_av = variables["X-averaged separator porosity"]
        eps_p_av = variables["X-averaged positive electrode porosity"]

        self.initial_conditions = {eps_n_av: self.param.epsilon_n_init}
        self.initial_conditions.update({eps_s_av: self.param.epsilon_s_init})
        self.initial_conditions.update({eps_p_av: self.param.epsilon_p_init})
