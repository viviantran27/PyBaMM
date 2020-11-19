#
# Base class for particles
#
import pybamm


class BaseParticle(pybamm.BaseSubModel):
    """
    Base class for molar conservation in particles.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_standard_concentration_variables(
        self, c_s, c_s_xav=None, c_s_rav=None, c_s_av=None, c_s_surf=None
    ):
        """
        All particle submodels must provide the particle concentration as an argument
        to this method. Some submodels solve for quantities other than the concentration
        itself, for example the 'FickianSingleParticle' models solves for the x-averaged
        concentration. In such cases the variables being solved for (set in
        'get_fundamental_variables') must also be passed as keyword arguments. If not
        passed as keyword arguments, the various average concentrations and surface
        concentration are computed automatically from the particle concentration.
        """

        # Get surface concentration if not provided as fundamental variable to
        # solve for
        c_s_surf = c_s_surf or pybamm.surf(c_s)
        c_s_surf_av = pybamm.x_average(c_s_surf)

        if self.domain == "Negative":
            c_scale = self.param.c_n_max
            eps_s = self.param.epsilon_s_n
            L = self.param.L_n
        elif self.domain == "Positive":
            c_scale = self.param.c_p_max
            eps_s = self.param.epsilon_s_p
            L = self.param.L_p
        A = self.param.A_cc

        # Get average concentration(s) if not provided as fundamental variable to
        # solve for
        c_s_xav = c_s_xav or pybamm.x_average(c_s)
        c_s_rav = c_s_rav or pybamm.r_average(c_s)
        c_s_av = c_s_av or pybamm.r_average(c_s_xav)
        c_s_vol_av = pybamm.x_average(eps_s * c_s_rav)

        variables = {
            self.domain + " particle concentration": c_s,
            self.domain + " particle concentration [mol.m-3]": c_s * c_scale,
            self.domain + " particle concentration [mol.m-3]": c_s * c_scale,
            "X-averaged " + self.domain.lower() + " particle concentration": c_s_xav,
            "X-averaged "
            + self.domain.lower()
            + " particle concentration [mol.m-3]": c_s_xav * c_scale,
            "R-averaged " + self.domain.lower() + " particle concentration": c_s_rav,
            "R-averaged "
            + self.domain.lower()
            + " particle concentration [mol.m-3]": c_s_rav * c_scale,
            "Average " + self.domain.lower() + " particle concentration": c_s_av,
            "Average "
            + self.domain.lower()
            + " particle concentration [mol.m-3]": c_s_av * c_scale,
            self.domain + " particle surface concentration": c_s_surf,
            self.domain
            + " particle surface concentration [mol.m-3]": c_scale * c_s_surf,
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration": c_s_surf_av,
            "X-averaged "
            + self.domain.lower()
            + " particle surface concentration [mol.m-3]": c_scale * c_s_surf_av,
            self.domain + " electrode active volume fraction": eps_s,
            self.domain + " electrode volume-averaged concentration": c_s_vol_av,
            self.domain
            + " electrode "
            + "volume-averaged concentration [mol.m-3]": c_s_vol_av * c_scale,
            self.domain + " electrode extent of lithiation": c_s_rav,
            "X-averaged "
            + self.domain.lower()
            + " electrode extent of lithiation": c_s_av,
            "Total lithium in "
            + self.domain.lower()
            + " electrode [mol]": c_s_vol_av * c_scale * L * A,
            "Minimum "
            + self.domain.lower()
            + " particle concentration": pybamm.min(c_s),
            "Maximum "
            + self.domain.lower()
            + " particle concentration": pybamm.max(c_s),
            "Minimum "
            + self.domain.lower()
            + " particle concentration [mol.m-3]": pybamm.min(c_s) * c_scale,
            "Maximum "
            + self.domain.lower()
            + " particle concentration [mol.m-3]": pybamm.max(c_s) * c_scale,
            "Minimum "
            + self.domain.lower()
            + " particle surface concentration": pybamm.min(c_s_surf),
            "Maximum "
            + self.domain.lower()
            + " particle surface concentration": pybamm.max(c_s_surf),
            "Minimum "
            + self.domain.lower()
            + " particle surface concentration [mol.m-3]": pybamm.min(c_s_surf)
            * c_scale,
            "Maximum "
            + self.domain.lower()
            + " particle surface concentration [mol.m-3]": pybamm.max(c_s_surf)
            * c_scale,
        }

        variables.update(self._get_microstrcuture_variables())

        return variables

    def _get_microstrcuture_variables(self):
        if self.domain == "Negative":
            x = pybamm.standard_spatial_vars.x_n
            R = self.param.R_n(x)
            R_scale = self.param.R_n_typ
            a = self.param.a_n(x)
            a_scale = self.param.a_n_typ
        elif self.domain == "Positive":
            x = pybamm.standard_spatial_vars.x_p
            R = self.param.R_p(x)
            R_scale = self.param.R_p_typ
            a = self.param.a_p(x)
            a_scale = self.param.a_p_typ

        variables = {
            self.domain + " particle radius": R,
            self.domain + " particle radius [m]": R * R_scale,
            self.domain + " electrode surface area per unit volume": a,
            self.domain + " electrode surface area per unit volume [m-1]": a * a_scale,
        }

        return variables

    def _get_standard_flux_variables(self, N_s, N_s_xav):
        variables = {
            self.domain + " particle flux": N_s,
            "X-averaged " + self.domain.lower() + " particle flux": N_s_xav,
        }

        return variables

    def set_events(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        tol = 1e-4

        self.events.append(
            pybamm.Event(
                "Minumum " + self.domain.lower() + " particle surface concentration",
                pybamm.min(c_s_surf) - tol,
                pybamm.EventType.TERMINATION,
            )
        )

        self.events.append(
            pybamm.Event(
                "Maximum " + self.domain.lower() + " particle surface concentration",
                (1 - tol) - pybamm.max(c_s_surf),
                pybamm.EventType.TERMINATION,
            )
        )
