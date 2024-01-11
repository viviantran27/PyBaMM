#
# Base class for thermal effects
#
import pybamm
import numpy as np


class BaseThermal(pybamm.BaseSubModel):
    """
    Base class for thermal effects

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param, options=options)

    def _get_standard_fundamental_variables(
        self, T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
    ):
        """
        Note: here we explicitly pass in the averages for the temperature as computing
        the average temperature in `BaseThermal` using `self._x_average` requires a
        workaround to avoid raising a `ModelError` (as the key in the equation
        dict gets modified).

        For more information about this method in general,
        see :meth:`pybamm.base_submodel._get_standard_fundamental_variables`
        """
        param = self.param

        # The variable T is the concatenation of the temperature in the negative
        # electrode, separator and positive electrode, for use in the electrochemical
        # models
        T = pybamm.concatenation(T_n, T_s, T_p)

        # Compute averaged temperatures by domain
        if self.half_cell:
            # overwrite T_n to be the boundary value of T_s
            T_n = pybamm.boundary_value(T_s, "left")
        T_n_av = pybamm.x_average(T_n)
        T_s_av = pybamm.x_average(T_s)
        T_p_av = pybamm.x_average(T_p)

        # Get the ambient temperature, which can be specified as a function of time
        T_amb_dim = param.T_amb_dim(pybamm.t * param.timescale)
        T_amb = param.T_amb(pybamm.t * param.timescale)

        variables = {
            "Negative current collector temperature": T_cn,
            "Negative current collector temperature [K]": param.Delta_T * T_cn
            + param.T_ref,
            "X-averaged negative electrode temperature": T_n_av,
            "X-averaged negative electrode temperature [K]": param.Delta_T * T_n_av
            + param.T_ref,
            "Negative electrode temperature": T_n,
            "Negative electrode temperature [K]": param.Delta_T * T_n + param.T_ref,
            "X-averaged separator temperature": T_s_av,
            "X-averaged separator temperature [K]": param.Delta_T * T_s_av
            + param.T_ref,
            "Separator temperature": T_s,
            "Separator temperature [K]": param.Delta_T * T_s + param.T_ref,
            "X-averaged positive electrode temperature": T_p_av,
            "X-averaged positive electrode temperature [K]": param.Delta_T * T_p_av
            + param.T_ref,
            "Positive electrode temperature": T_p,
            "Positive electrode temperature [K]": param.Delta_T * T_p + param.T_ref,
            "Positive current collector temperature": T_cp,
            "Positive current collector temperature [K]": param.Delta_T * T_cp
            + param.T_ref,
            "Cell temperature": T,
            "Cell temperature [K]": param.Delta_T * T + param.T_ref,
            "X-averaged cell temperature": T_x_av,
            "X-averaged cell temperature [K]": param.Delta_T * T_x_av + param.T_ref,
            "Volume-averaged cell temperature": T_vol_av,
            "Volume-averaged cell temperature [K]": param.Delta_T * T_vol_av + param.T_ref,
            "Ambient temperature [K]": T_amb_dim,
            "Ambient temperature": T_amb,
        }

        return variables

    def _get_standard_coupled_variables(self, variables):
        param = self.param

        # Tab heating 
        Q_scale = param.i_typ * param.potential_scale / param.L_x # moved to accommodate tabbing I^2R
        I = variables["Current [A]"]
        R_tab = pybamm.Parameter("Tabbing resistance [Ohm]")
        Q_tab = (I**2*R_tab/(param.L_x*param.L_y*param.L_z))/Q_scale # originally W.m-3

        Q_tabbing = pybamm.concatenation(   
            *[
                pybamm.FullBroadcast(Q_tab, ["negative electrode"], "current collector"),
                pybamm.FullBroadcast(Q_tab, ["separator"], "current collector"),
                pybamm.FullBroadcast(Q_tab, ["positive electrode"], "current collector"),
            ]
        ) #*param.geo.n.l

        # Ohmic heating in solid
        i_s_p = variables["Positive electrode current density"]
        phi_s_p = variables["Positive electrode potential"]
        Q_ohm_s_cn, Q_ohm_s_cp = self._current_collector_heating(variables)
        if self.half_cell:
            i_boundary_cc = variables["Current collector current density"]
            T_n = variables["Negative electrode temperature"]
            Q_ohm_s_n_av = i_boundary_cc ** 2 / param.n.sigma(T_n)
            Q_ohm_s_n = pybamm.PrimaryBroadcast(Q_ohm_s_n_av, "negative electrode")
        else:
            i_s_n = variables["Negative electrode current density"]
            phi_s_n = variables["Negative electrode potential"]
            Q_ohm_s_n = -pybamm.inner(i_s_n, pybamm.grad(phi_s_n))
        Q_ohm_s_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        Q_ohm_s_p = -pybamm.inner(i_s_p, pybamm.grad(phi_s_p))
        Q_ohm_s = pybamm.concatenation(Q_ohm_s_n, Q_ohm_s_s, Q_ohm_s_p)

        # Ohmic heating in electrolyte
        # TODO: change full stefan-maxwell conductivity so that i_e is always
        # a Concatenation
        i_e = variables["Electrolyte current density"]
        phi_e = variables["Electrolyte potential"]
        if isinstance(i_e, pybamm.Concatenation):
            # compute by domain if possible
            phi_e_s = variables["Separator electrolyte potential"]
            phi_e_p = variables["Positive electrolyte potential"]
            if self.half_cell:
                i_e_s, i_e_p = i_e.orphans
                Q_ohm_e_n = pybamm.FullBroadcast(
                    0, ["negative electrode"], "current collector"
                )
            else:
                i_e_n, i_e_s, i_e_p = i_e.orphans
                phi_e_n = variables["Negative electrolyte potential"]
                Q_ohm_e_n = -pybamm.inner(i_e_n, pybamm.grad(phi_e_n))
            Q_ohm_e_s = -pybamm.inner(i_e_s, pybamm.grad(phi_e_s))
            Q_ohm_e_p = -pybamm.inner(i_e_p, pybamm.grad(phi_e_p))
            Q_ohm_e = pybamm.concatenation(Q_ohm_e_n, Q_ohm_e_s, Q_ohm_e_p)
        else:
            # else compute using i_e across all domains
            if self.half_cell:
                Q_ohm_e_n = pybamm.FullBroadcast(
                    0, ["negative electrode"], "current collector"
                )
                Q_ohm_e_s_p = -pybamm.inner(i_e, pybamm.grad(phi_e))
                Q_ohm_e = pybamm.concatenation(Q_ohm_e_n, Q_ohm_e_s_p)
            else:
                Q_ohm_e = -pybamm.inner(i_e, pybamm.grad(phi_e))

        # Total Ohmic heating
        Q_ohm = Q_ohm_s + Q_ohm_e 

        # Side reaction heating
        Q_decomp_an = variables["Anode decomposition heating"]
        Q_decomp_sei = variables["SEI decomposition heating"]
        Q_decomp_ca = variables["Cathode decomposition heating"]
        Q_decomp_n = Q_decomp_an + Q_decomp_sei
        Q_decomp_p = Q_decomp_ca

        # Irreversible electrochemical heating
        a_p = variables["Positive electrode surface area to volume ratio"]
        j_p = variables["Positive electrode interfacial current density"]
        eta_r_p = variables["Positive electrode reaction overpotential"]
        if self.half_cell:
            Q_rxn_n = pybamm.FullBroadcast(
                0, ["negative electrode"], "current collector"
            )
        else:
            a_n = variables["Negative electrode surface area to volume ratio"]
            j_n = variables["Negative electrode interfacial current density"]
            eta_r_n = variables["Negative electrode reaction overpotential"]
            Q_rxn_n = a_n * j_n * eta_r_n + Q_decomp_n
        Q_rxn_p = a_p * j_p * eta_r_p + Q_decomp_p
        Q_rxn = pybamm.concatenation(   
            *[
                Q_rxn_n,
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                Q_rxn_p,
            ]
        )

        # Reversible electrochemical heating
        T_p = variables["Positive electrode temperature"]
        dUdT_p = variables["Positive electrode entropic change"]
        if self.half_cell:
            Q_rev_n = pybamm.FullBroadcast(
                0, ["negative electrode"], "current collector"
            )
        else:
            T_n = variables["Negative electrode temperature"]
            dUdT_n = variables["Negative electrode entropic change"]
            Q_rev_n = a_n * j_n * (param.Theta ** (-1) + T_n) * dUdT_n
        Q_rev_p = a_p * j_p * (param.Theta ** (-1) + T_p) * dUdT_p
        Q_rev = pybamm.concatenation(
            *[
                Q_rev_n,
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                Q_rev_p,
            ]
        )

        # Particle concentration overpotential 
        Un =  param.n.prim.U
        c_n_bulk = variables['Average negative particle concentration']
        Un_c_surf = variables['Negative electrode open circuit potential']
        Q_mix_n = j_n*(Un_c_surf-Un(c_n_bulk,T_n))
        Up = param.p.prim.U
        c_p_bulk = variables['Average positive particle concentration']
        Up_c_surf = variables['Positive electrode open circuit potential']
        Q_mix_p = j_p*(Up_c_surf-Up(c_p_bulk,T_p))
        Q_mix = pybamm.concatenation(
            *[
                Q_mix_n,
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                Q_mix_p,
            ]
        )

        # Heat of mixing
        # Q_mix_s_n, Q_mix_s_s, Q_mix_s_p = self._heat_of_mixing(variables)
        # Q_mix = pybamm.concatenation(Q_mix_s_n, Q_mix_s_s, Q_mix_s_p)
        # Q_mix = pybamm.concatenation(
        #     *[
        #         pybamm.FullBroadcast(Q_mix_s_n,["negative electrode"], "current collector"),
        #         Q_mix_s_s,
        #         Q_mix_s_p,
        #     ]
        # )

        # Total heating
        Q = Q_ohm + Q_rxn + Q_rev + Q_tabbing + Q_mix

        # Compute the X-average over the entire cell, including current collectors
        Q_tab_av = self._x_average(Q_tabbing, 0, 0)
        Q_ohm_av = self._x_average(Q_ohm, Q_ohm_s_cn, Q_ohm_s_cp)
        Q_rxn_av = self._x_average(Q_rxn, 0, 0)
        Q_rev_av = self._x_average(Q_rev, 0, 0)
        Q_mix_av = self._x_average(Q_mix, 0, 0)
        Q_av = self._x_average(Q, Q_ohm_s_cn, Q_ohm_s_cp)

        # Compute volume-averaged heat source terms 
        Q_tab_vol_av = self._yz_average(Q_tab_av)
        Q_ohm_vol_av = self._yz_average(Q_ohm_av)
        Q_rxn_vol_av = self._yz_average(Q_rxn_av)
        Q_rev_vol_av = self._yz_average(Q_rev_av)
        Q_mix_vol_av = self._yz_average(Q_mix_av)
        Q_vol_av = self._yz_average(Q_av)

        # Dimensional scaling for heat source terms
        Q_scale = param.i_typ * param.potential_scale / param.L_x

        variables.update(
            {
                # "Test heat": Q_test,
                "Ohmic heating": Q_ohm,
                "Volume-averaged tab heating": Q_tab_vol_av,
                "Ohmic heating [W.m-3]": Q_ohm * Q_scale,
                "X-averaged Ohmic heating": Q_ohm_av,
                "X-averaged Ohmic heating [W.m-3]": Q_ohm_av * Q_scale,
                "Volume-averaged Ohmic heating": Q_ohm_vol_av,
                "Volume-averaged Ohmic heating [W.m-3]": Q_ohm_vol_av * Q_scale,
                "Irreversible electrochemical heating": Q_rxn,
                "Irreversible electrochemical heating [W.m-3]": Q_rxn * Q_scale,
                "X-averaged irreversible electrochemical heating": Q_rxn_av,
                "X-averaged irreversible electrochemical heating [W.m-3]": Q_rxn_av
                * Q_scale,
                "Volume-averaged irreversible electrochemical heating": Q_rxn_vol_av,
                "Volume-averaged irreversible electrochemical heating "
                + "[W.m-3]": Q_rxn_vol_av * Q_scale,
                "Reversible heating": Q_rev,
                "Reversible heating [W.m-3]": Q_rev * Q_scale,
                "X-averaged reversible heating": Q_rev_av,
                "X-averaged reversible heating [W.m-3]": Q_rev_av * Q_scale,
                "Volume-averaged reversible heating": Q_rev_vol_av,
                "Volume-averaged reversible heating [W.m-3]": Q_rev_vol_av * Q_scale,
                "Heat of mixing": Q_mix,
                "Heat of mixing [W.m-3]": Q_mix * Q_scale,
                "X-averaged heat of mixing": Q_mix_av,
                "X-averaged heat of mixing [W.m-3]": Q_mix_av * Q_scale,
                "Volume-averaged heat of mixing": Q_mix_vol_av,
                "Volume-averaged heat of mixing [W.m-3]": Q_mix_vol_av * Q_scale,
                "Total heating": Q,
                "Total heating [W.m-3]": Q * Q_scale,
                "X-averaged total heating": Q_av,
                "X-averaged total heating [W.m-3]": Q_av * Q_scale,
                "Volume-averaged total heating": Q_vol_av,
                "Volume-averaged total heating [W.m-3]": Q_vol_av * Q_scale,
                "Tab heating [W.m-3]": Q_tab_vol_av * Q_scale,
                "Electrochemical Ohmic heating [W.m-3]": (Q_ohm_s + Q_ohm_e) * Q_scale,
                "X-averaged electrochemical Ohmic heating": self._x_average(Q_ohm_s + Q_ohm_e, Q_ohm_s_cn, Q_ohm_s_cp),
                "X-averaged electrochemical Ohmic heating [W.m-3]": self._x_average(Q_ohm_s + Q_ohm_e, Q_ohm_s_cn, Q_ohm_s_cp)* Q_scale,
                "Volume-averaged electrochemical Ohmic heating": self._yz_average(self._x_average(Q_ohm_s + Q_ohm_e, Q_ohm_s_cn, Q_ohm_s_cp)),
                "Volume-averaged electrochemical Ohmic heating [W.m-3]": self._yz_average(self._x_average(Q_ohm_s + Q_ohm_e, Q_ohm_s_cn, Q_ohm_s_cp)) * Q_scale,
            }
        )
        return variables

    def _heat_of_mixing(self, variables):
        """Compute heat of mixing source terms."""
        param = self.param

        if self.options["heat of mixing"] == "true":
            F = pybamm.constants.F.value
            pi = np.pi

            # Compute heat of mixing in negative electrode
            # if self.options.electrode_types["negative"] == "planar":
            #     Q_mix_s_n = pybamm.FullBroadcast(
            #         0, ["negative electrode"], "current collector"
            #     )
            # else:
            a_n = variables["Negative electrode surface area to volume ratio [m-1]"]
            R_n = variables["Negative particle radius [m]"]
            N_n = a_n / (4 * pi * R_n**2)

            c_n = variables["X-averaged negative particle concentration [mol.m-3]"]
            T_n = variables["X-averaged negative electrode temperature [K]"]
            
            T_n_part = pybamm.PrimaryBroadcast(T_n, ["negative particle"])

            # c_n_part = pybamm.FullBroadcast(c_n , ["negative particle"], 'current collector')
            D_n = param.n.prim.D_dimensional(c_n/param.n.prim.c_max, T_n_part)
            dc_n_dr2 = pybamm.inner(pybamm.grad(c_n),pybamm.grad(c_n))
            
            # dUeq_n = 1 #param.n.prim.dUdsto_dimensional(c_n / param.n.prim.c_max, T_n_part)
            # print(dUeq_n.shape_for_testing)
            # dUeq_n = pybamm.FullBroadcast(dUeq_n, ["negative particle"], 'current collector')
            # Ueq_n = param.n.prim.U_dimensional
            # TODO: Drop terms as it has spatial operators and diff doesn't work
            dUeq_n = param.n.prim.dUdsto_dimensional(c_n/ param.n.prim.c_max, T_n_part) #children[0].children[0]#
            # print(dUeq_n.shape_for_testing) #165 = 5*33

            integrand_r_n = (D_n * dc_n_dr2 * dUeq_n/ param.n.prim.c_max)# domain: negative particle
            # integrand_r_n = integrand_r_n.create_copy()
            # integrand_r_n.domains.update({'secondary':'negative electrode'})
            print(integrand_r_n.shape_for_testing) #165 = 5*33
            print(integrand_r_n.domain) #165 = 5*33

            integration_variable_r_n = pybamm.SpatialVariable("r", 
                domain=integrand_r_n.domain, 
                coord_sys="spherical polar")
            # print(integration_variable_r_n.shape_for_testing) #5
            # integration_variable_r_n = pybamm.standard_spatial_vars.r_n
            print(integration_variable_r_n.shape_for_testing) #165
            print(integration_variable_r_n.domains) 
            integral_r_n = pybamm.Integral(integrand_r_n, integration_variable_r_n) # domain: negative electrode
            print(integral_r_n.shape_for_testing) # 33
            print(integral_r_n.domains) # 33
            
            integral_r_n_electrode =  pybamm.SecondaryBroadcast(integral_r_n, ["negative electrode"])
            Q_mix_s_n = -F * N_n * integral_r_n_electrode # domain: negative electrode
            # print(Q_mix_s_n.shape_for_testing) # 33
            # print(Q_mix_s_n.shape_for_testing)
            # print(Q_mix_s_n.evaluate_for_shape_using_domain)
            # Q_mix_s_n = pybamm.FullBroadcast(
            #     5, ["negative electrode"], "current collector"
            # )

            # # Compute heat of mixing in positive electrode
            # a_p = variables["Positive electrode surface area to volume ratio [m-1]"]
            # R_p = variables["Positive particle radius [m]"]
            # N_p = a_p / (4 * pi * R_p**2)
            # # if self.x_average:
            # #     c_p = variables["X-averaged positive particle concentration [mol.m-3]"]
            # #     T_p = variables["X-averaged positive electrode temperature [K]"]
            # # else:
            # c_p = variables["Positive particle concentration [mol.m-3]"]
            # T_p = variables["Positive electrode temperature [K]"]
            # T_p_part = pybamm.PrimaryBroadcast(T_p, ["positive particle"])
            # dc_p_dr2 = pybamm.inner(pybamm.grad(c_p), pybamm.grad(c_p))
            # D_p = param.p.prim.D(c_p, T_p_part)
            # dUeq_p = param.p.prim.dUdsto_dimensional(c_p / param.p.prim.c_max, T_p_part)
            # integrand_r_p = D_p * dc_p_dr2 * dUeq_p / param.p.prim.c_max
            # integration_variable_r_p = [
            #     pybamm.SpatialVariable("r", domain=integrand_r_p.domain)
            # ]
            # integral_r_p = pybamm.Integral(integrand_r_p, integration_variable_r_p)
            # Q_mix_s_p = -F * N_p * integral_r_p
            # Q_mix_s_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        else:
            Q_mix_s_n = pybamm.FullBroadcast(
                0, ["negative electrode"], "current collector"
            )
        # print(Q_mix_s_n.shape_for_testing) #33
        Q_mix_s_p = pybamm.FullBroadcast(
            0, ["positive electrode"], "current collector"
        )
        Q_mix_s_s = pybamm.FullBroadcast(
            0, ["separator"], "current collector"
        )
        return Q_mix_s_n, Q_mix_s_s, Q_mix_s_p

    def _x_average(self, var, var_cn, var_cp):
        """
        Computes the X-average over the whole cell (including current collectors)
        from the variable in the cell (negative electrode, separator,
        positive electrode), negative current collector, and positive current
        collector.
        Note: we do this as we cannot create a single variable which is
        the concatenation [var_cn, var, var_cp] since var_cn and var_cp share the
        same domain. (In the N+1D formulation the current collector variables are
        assumed independent of x, so we do not make the distinction between negative
        and positive current collectors in the geometry).
        """
        out = (
            self.param.n.L_cc * var_cn
            + self.param.L_x * pybamm.x_average(var)
            + self.param.p.L_cc * var_cp
        ) / self.param.L
        return out

    def _yz_average(self, var):
        """Computes the y-z average."""
        # TODO: change the behaviour of z_average and yz_average so the if statement
        # can be removed
        if self.options["dimensionality"] in [0, 1]:
            return pybamm.z_average(var)
        elif self.options["dimensionality"] == 2:
            return pybamm.yz_average(var)
    
    def _current_collector_heating(self, variables):
        """Compute Ohmic heating in current collectors."""
        cc_dimension = self.options["dimensionality"]

        # Compute the Ohmic heating for 0D current collectors
        if cc_dimension == 0:
            i_boundary_cc = variables["Current collector current density"]
            Q_s_cn = i_boundary_cc ** 2 / self.param.n.sigma_cc
            Q_s_cp = i_boundary_cc ** 2 / self.param.p.sigma_cc
        # Otherwise we compute the Ohmic heating for 1 or 2D current collectors
        # In this limit the current flow is all in the y,z direction in the current
        # collectors
        elif cc_dimension in [1, 2]:
            phi_s_cn = variables["Negative current collector potential"]
            phi_s_cp = variables["Positive current collector potential"]
            # TODO: implement grad_squared in other spatial methods so that the
            # if statement can be removed
            if cc_dimension == 1:
                Q_s_cn = self.param.n.sigma_cc_prime * pybamm.inner(
                    pybamm.grad(phi_s_cn), pybamm.grad(phi_s_cn)
                )
                Q_s_cp = self.param.p.sigma_cc_prime * pybamm.inner(
                    pybamm.grad(phi_s_cp), pybamm.grad(phi_s_cp)
                )
            elif cc_dimension == 2:
                # Inner not implemented in 2D -- have to call grad_squared directly
                Q_s_cn = self.param.n.sigma_cc_prime * pybamm.grad_squared(phi_s_cn)
                Q_s_cp = self.param.p.sigma_cc_prime * pybamm.grad_squared(phi_s_cp)
        return Q_s_cn, Q_s_cp

    def _x_average(self, var, var_cn, var_cp):
        """
        Computes the X-average over the whole cell (including current collectors)
        from the variable in the cell (negative electrode, separator,
        positive electrode), negative current collector, and positive current
        collector.
        Note: we do this as we cannot create a single variable which is
        the concatenation [var_cn, var, var_cp] since var_cn and var_cp share the
        same domain. (In the N+1D formulation the current collector variables are
        assumed independent of x, so we do not make the distinction between negative
        and positive current collectors in the geometry).
        """
        out = (
            self.param.n.l_cc * var_cn
            + self.param.l_x * pybamm.x_average(var)
            + self.param.p.l_cc * var_cp
        ) / self.param.l
        return out

    def _yz_average(self, var):
        """Computes the y-z average."""
        # TODO: change the behaviour of z_average and yz_average so the if statement
        # can be removed
        if self.options["dimensionality"] in [0, 1]:
            return pybamm.z_average(var)
        elif self.options["dimensionality"] == 2:
            return pybamm.yz_average(var)
