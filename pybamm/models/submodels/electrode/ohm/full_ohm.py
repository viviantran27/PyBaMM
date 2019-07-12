#
# Full model of electrode employing Ohm's law
#
import pybamm
from .base_ohm import BaseModel


class Full(BaseModel):
    """Full model of electrode employing Ohm's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative' or 'Positive'


    **Extends:** :class:`pybamm.electrode.ohm.BaseModel`
    """

    def __init__(self, param, domain, reactions, options=None):
        super().__init__(param, domain, reactions)

        if options:
            self.options = options
        else:
            self.options = {"probelem type": "galvanostatic"}

    def get_fundamental_variables(self):

        if self.domain == "Negative":
            phi_s = pybamm.standard_variables.phi_s_n
        elif self.domain == "Positive":
            phi_s = pybamm.standard_variables.phi_s_p

        variables = self._get_standard_potential_variables(phi_s)

        return variables

    def get_coupled_variables(self, variables):

        phi_s = variables[self.domain + " electrode potential"]
        eps = variables[self.domain + " electrode porosity"]

        if self.domain == "Negative":
            sigma = self.param.sigma_n
        elif self.domain == "Positive":
            sigma = self.param.sigma_p

        sigma_eff = sigma * (1 - eps) ** self.param.b
        i_s = -sigma_eff * pybamm.grad(phi_s)

        variables.update({self.domain + " electrode effective conductivity": sigma_eff})

        variables.update(self._get_standard_current_variables(i_s))

        if self.domain == "Positive":
            variables.update(self._get_standard_whole_cell_current_variables(variables))

        if (
            self.domain == "Positive"
            and self.options["problem type"] == "potentiostatic"
        ):
            variables.update(self._get_current_collector_current_density(i_s))

        return variables

    def _get_current_collector_current_density(self, i_s):
        i_boundary_cc = pybamm.BoundaryValue(i_s, "right")
        variables = {"Current collector current density": i_boundary_cc}
        return variables

    def set_algebraic(self, variables):

        phi_s = variables[self.domain + " electrode potential"]
        i_s = variables[self.domain + " electrode current density"]
        sum_j = sum(
            variables[reaction[self.domain]["aj"]]
            for reaction in self.reactions.values()
        )

        self.algebraic[phi_s] = pybamm.div(i_s) + sum_j

    def set_boundary_conditions(self, variables):

        if self.options["problem type"] == "galvanostatic":
            self._set_galvanostatic_boundary_conditions(variables)
        elif self.options["problem type"] == "potentiostatic":
            self._set_potentiostatic_boundary_conditions(variables)
        else:
            raise pybamm.OptionError(
                """The option 'problem type' must be either: 'galvanostatic'
                or 'potentiostatic' and not {}.""".format(
                    self.options["problem type"]
                )
            )

    def _set_galvanostatic_boundary_conditions(self, variables):
        phi_s = variables[self.domain + " electrode potential"]
        eps = variables[self.domain + " electrode porosity"]
        i_boundary_cc = variables["Current collector current density"]

        if self.domain == "Negative":
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "Positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            sigma_eff = self.param.sigma_p * (1 - eps) ** self.param.b
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-sigma_eff, "right"),
                "Neumann",
            )

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}

    def _set_potentiostatic_boundary_conditions(self, variables):

        phi_s = variables[self.domain + " electrode potential"]
        v_cc = variables["Local current collector potential difference"]

        if self.domain == "Negative":
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self.domain == "Positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            rbc = (v_cc, "Dirichlet")

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}

    def set_initial_conditions(self, variables):

        phi_s = variables[self.domain + " electrode potential"]

        if self.domain == "Negative":
            phi_s_init = pybamm.Scalar(0)
        elif self.domain == "Positive":
            phi_s_init = self.param.U_p(self.param.c_p_init) - self.param.U_n(
                self.param.c_n_init
            )

        self.initial_conditions[phi_s] = phi_s_init

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
