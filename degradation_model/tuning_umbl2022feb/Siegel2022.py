import pybamm
import os
import numpy as np

def graphite_diffusivity_PeymanMPM(sto, T):
    """
    Graphite diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Peyman MPM.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 5.0 * 10 ** (-15)
    E_D_s = 42770
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def graphite_ocp_PeymanMPM(sto):
    """
    Graphite Open-circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].

    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)
    """

    u_eq = (
        0.063
        + 0.8 * pybamm.exp(-75 * (sto + 0.001))
        - 0.0120 * pybamm.tanh((sto - 0.127) / 0.016)
        - 0.0118 * pybamm.tanh((sto - 0.155) / 0.016)
        - 0.0035 * pybamm.tanh((sto - 0.220) / 0.020)
        - 0.0095 * pybamm.tanh((sto - 0.190) / 0.013)
        - 0.0145 * pybamm.tanh((sto - 0.490) / 0.020)
        - 0.0800 * pybamm.tanh((sto - 1.030) / 0.055)
    )

    return u_eq


def graphite_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.
    Check the unit of Reaction rate constant k0 is from Peyman MPM.

    References
    ----------
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 1.061 * 10 ** (-6)  # unit has been converted
    # units are (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 37480
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def graphite_entropic_change_PeymanMPM(sto, c_s_max):
    """
    Graphite entropic change in open-circuit potential (OCP) at a temperature of
    298.15K as a function of the stochiometry taken from [1]

    References
    ----------
    .. [1] K.E. Thomas, J. Newman, "Heats of mixing and entropy in porous insertion
           electrode", J. of Power Sources 119 (2003) 844-849

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    du_dT = 10 ** (-3) * (
        0.28
        - 1.56 * sto
        - 8.92 * sto ** (2)
        + 57.21 * sto ** (3)
        - 110.7 * sto ** (4)
        + 90.71 * sto ** (5)
        - 27.14 * sto ** (6)
    )

    return du_dT


def NMC_diffusivity_PeymanMPM(sto, T):
    """
    NMC diffusivity as a function of stochiometry, in this case the
    diffusivity is taken to be a constant. The value is taken from Peyman MPM.

    References
    ----------
    .. [1] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_ref = 8 * 10 ** (-15)
    E_D_s = 18550
    arrhenius = pybamm.exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def NMC_ocp_Siegel2022(sto):
    """
    Nickel Managanese Cobalt Oxide (NMC) Open-circuit Potential (OCP) as a
    function of the stochiometry. The fit is taken from Peyman MPM.

    References
    ----------
    Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """
    u_eq = (
        -2708.586858 * (sto ** 9) +13443.432624 * (sto ** 8) 
        -28538.487067 * (sto ** 7) +33871.076514 * (sto ** 6) 
        -24665.620934 * (sto ** 5) +11381.032457 * (sto ** 4) 
        -3313.239062 * (sto ** 3) +586.285653 * (sto ** 2) 
        -59.774094 * (sto ** 1) +7.408439 
        )
    # u_eq = (
    #     4.3452
    #     - 1.6518 * sto
    #     + 1.6225 * (sto**2)
    #     - 2.0843 * (sto**3)
    #     + 3.5146 * (sto**4)
    #     - 2.2166 * (sto**5)
    #     - 0.5623e-4 * pybamm.exp(109.451 * sto - 100.006)
    # )

    return u_eq


def NMC_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.

    References
    ----------
    .. Peyman MPM manuscript (to be submitted)

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 4.824 * 10 ** (-6)  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 39570
    arrhenius = pybamm.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    )


def NMC_entropic_change_PeymanMPM(sto, c_s_max):
    """
    Nickel Manganese Cobalt (NMC) entropic change in open-circuit potential (OCP) at
    a temperature of 298.15K as a function of the OCP. The fit is taken from [1].

    References
    ----------
    .. [1] W. Le, I. Belharouak, D. Vissers, K. Amine, "In situ thermal study of
    li1+ x [ni1/ 3co1/ 3mn1/ 3] 1- x o2 using isothermal micro-clorimetric
    techniques",
    J. of the Electrochemical Society 153 (11) (2006) A2147-A2151.

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """

    # Since the equation uses the OCP at each stoichiometry as input,
    # we need OCP function here

    u_eq = (
        4.3452
        - 1.6518 * sto
        + 1.6225 * sto**2
        - 2.0843 * sto**3
        + 3.5146 * sto**4
        - 0.5623 * 10 ** (-4) * pybamm.exp(109.451 * sto - 100.006)
    )

    du_dT = (
        -800 + 779 * u_eq - 284 * u_eq**2 + 46 * u_eq**3 - 2.8 * u_eq**4
    ) * 10 ** (-3)

    return du_dT


def electrolyte_diffusivity_PeymanMPM(c_e, T):
    """
    Diffusivity of LiPF6 in EC:DMC as a function of ion concentration. The original data
    is from [1]. The fit from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity
    """

    D_c_e = 5.35 * 10 ** (-10)
    E_D_e = 37040
    arrhenius = pybamm.exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_PeymanMPM(c_e, T):
    """
    Conductivity of LiPF6 in EC:DMC as a function of ion concentration. The original
    data is from [1]. The fit is from Dualfoil [2].

    References
    ----------
    .. [1] C Capiglia et al. 7Li and 19F diffusion coefficients and thermal
    properties of non-aqueous electrolyte solutions for rechargeable lithium batteries.
    Journal of power sources 81 (1999): 859-862.
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html
    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte conductivity
    """

    sigma_e = 1.3
    E_k_e = 34700
    arrhenius = pybamm.exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


def electrolyte_diffusivity_Dix(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC vol 3:7 with 2 wt% VC as a function of
    ion concentration.
    The original data is from [1]. The Fitting Funciton is from [2]

    References
    ----------
    .. [1]  Sean T. Dix et al. "Leveraging Molecular Dynamics to Improve Porous
    Electrode Theory Modeling Predictions of Lithium-Ion Battery Cells." 2023
    J. Electrochem. Soc. 170 083503
    DOI 10.1149/1945-7111/aceca8
       [2] Lars Ole Valøen and Jan N. Reimers. "Transport Properties of LiPF6-Based
       Li-Ion Battery Electrolytes." 2005 J. Electrochem. Soc. 152 A882 Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte diffusivity
    """

    # Diffusivity in 10E-10 m^2/s = 0.0001*10^((D00+ D01 / (T -
    # ((Tg0 + Conc * Tg1))))+ (D10+ D11/ (T - ((Tg0 + Conc* Tg1)))) * Conc)
    # Where Conc is in units of M and T is in K

    D00 = -487.2
    D01 = -28530000
    Tg0 = 58320
    Tg1 = 50.31
    D10 = -52.79
    D11 = -3061000
    c_e_M = c_e / 1000
    D_c_e = (
        (10**-10)
        * 0.0001
        * 10
        ** (
            (D00 + D01 / (T - ((Tg0 + c_e_M * Tg1))))
            + (D10 + D11 / (T - ((Tg0 + c_e_M * Tg1)))) * c_e_M
        )
    )

    return D_c_e


def electrolyte_conductivity_Dix(c_e, T):
    """
    Conductivity of LiPF6 in 3:7 EC:DMC as a function of ion concentration. The original
    data is from [1].

    References
    ----------
    ..     .. [1]  Sean T. Dix et al. "Leveraging Molecular Dynamics to Improve
    Porous Electrode Theory Modeling Predictions of Lithium-Ion Battery Cells."
    2023 J. Electrochem. Soc. 170 083503 DOI 10.1149/1945-7111/aceca8.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte conductivity
    """
    k_Kmax = 3.58e-7
    k_Cmax = 4.84e-6
    k_a = 0.4793
    k_P2 = -0.3299
    k_P1 = 2.22e-14
    k_Ea = 13490
    k_Rg = 8.314
    k_T0 = 162.5
    # Conductivity in mS/cm = 1000*(k_Kmax* (Conc / K_Cmax) ^ k_a *
    # EXP(k_P2 * (Conc – k_Cmax) ^ 2 – k_P1 * (Conc- K_Cmax) / K_Cmax)
    # * EXP(-k_Ea / k_Rg * (1 / T - 1 / K_T0)))
    # Where Conc is in units of M and T is in K
    # convert to S/m
    c_e_M = c_e / 1000

    sigma_e = 1 * (
        k_Kmax
        * (c_e_M / k_Cmax) ** k_a
        * pybamm.exp(k_P2 * (c_e_M - k_Cmax) ** 2 - k_P1 * (c_e_M - k_Cmax) / k_Cmax)
        * pybamm.exp(-k_Ea / k_Rg * (1 / T - 1 / k_T0))
    )

    return sigma_e


def plating_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li plating reaction [A.m-2].
    References
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_e


def stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li stripping reaction [A.m-2].

    References
    ----------

    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.

    Parameters
    ----------

    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------

    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter("Lithium plating kinetic rate constant [m.s-1]")

    return pybamm.constants.F * k_plating * c_Li


def SEI_limited_dead_lithium_OKane2022(L_sei):
    """
    Decay rate for dead lithium formation [s-1].
    References
    ----------
    .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
    Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
    and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
    Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
    Parameters
    ----------
    L_sei : :class:`pybamm.Symbol`
        Total SEI thickness [m]
    Returns
    -------
    :class:`pybamm.Symbol`
        Dead lithium decay rate [s-1]
    """

    gamma_0 = pybamm.Parameter("Dead lithium decay constant [s-1]")
    L_inner_0 = pybamm.Parameter("Initial inner SEI thickness [m]")
    L_outer_0 = pybamm.Parameter("Initial outer SEI thickness [m]")
    L_sei_0 = L_inner_0 + L_outer_0

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma


def graphite_volume_change_Ai2020(sto, c_s_max):
    """
    Graphite particle volume change as a function of stochiometry [1, 2].

    References
    ----------
     .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] Rieger, B., Erhard, S. V., Rumpf, K., & Jossen, A. (2016).
     A new method to model the thickness change of a commercial pouch cell
     during discharge. Journal of The Electrochemical Society, 163(8), A1566-A1575.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry, dimensionless
        should be R-averaged particle concentration
    Returns
    -------
    t_change:class:`pybamm.Symbol`
        volume change, dimensionless, normalised by particle volume
    """
    p1 = 145.907
    p2 = -681.229
    p3 = 1334.442
    p4 = -1415.710
    p5 = 873.906
    p6 = -312.528
    p7 = 60.641
    p8 = -5.706
    p9 = 0.386
    p10 = -4.966e-05
    t_change = (
        p1 * sto**9
        + p2 * sto**8
        + p3 * sto**7
        + p4 * sto**6
        + p5 * sto**5
        + p6 * sto**4
        + p7 * sto**3
        + p8 * sto**2
        + p9 * sto
        + p10
    )
    return t_change


def graphite_cracking_rate_Ai2020(T_dim):
    """
    Graphite particle cracking rate as a function of temperature [1, 2].

    References
    ----------
     .. [1] Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
     Battery cycle life prediction with coupled chemical degradation and fatigue
     mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    Parameters
    ----------
    T_dim: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = 3.9e-20
    Eac_cr = 0  # to be implemented
    arrhenius = np.exp(Eac_cr / pybamm.constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius



def volume_change_Ai2020(sto, c_s_max):
    """
    Particle volume change as a function of stochiometry [1, 2].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Rieger, B., Erhard, S. V., Rumpf, K., & Jossen, A. (2016).
     A new method to model the thickness change of a commercial pouch cell
     during discharge. Journal of The Electrochemical Society, 163(8), A1566-A1575.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry, dimensionless
        should be R-averaged particle concentration
    Returns
    -------
    t_change:class:`pybamm.Symbol`
        volume change, dimensionless, normalised by particle volume
    """
    omega = pybamm.Parameter("Positive electrode partial molar volume [m3.mol-1]")
    t_change = omega * c_s_max * sto
    return t_change


def cracking_rate_Ai2020(T_dim):
    """
    Particle cracking rate as a function of temperature [1, 2].

    References
    ----------
     .. [1] > Ai, W., Kraft, L., Sturm, J., Jossen, A., & Wu, B. (2020).
     Electrochemical Thermal-Mechanical Modelling of Stress Inhomogeneity in
     Lithium-Ion Pouch Cells. Journal of The Electrochemical Society, 167(1), 013512
      DOI: 10.1149/2.0122001JES.
     .. [2] > Deshpande, R., Verbrugge, M., Cheng, Y. T., Wang, J., & Liu, P. (2012).
     Battery cycle life prediction with coupled chemical degradation and fatigue
     mechanics. Journal of the Electrochemical Society, 159(10), A1730.

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        temperature, [K]

    Returns
    -------
    k_cr: :class:`pybamm.Symbol`
        cracking rate, [m/(Pa.m0.5)^m_cr]
        where m_cr is another Paris' law constant
    """
    k_cr = 3.9e-20
    Eac_cr = 0  # to be implemented
    arrhenius = np.exp(Eac_cr / pybamm.constants.R * (1 / T_dim - 1 / 298.15))
    return k_cr * arrhenius



# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a graphite/NMC532 pouch cell from the paper

        Peyman Mohtat, Suhak Lee, Valentin Sulzer, Jason B. Siegel, and Anna G.
        Stefanopoulou. Differential Expansion and Voltage Model for Li-ion Batteries at
        Practical Charging Rates. Journal of The Electrochemical Society,
        167(11):110561, 2020. doi:10.1149/1945-7111/aba5d1.

    and references therein.

    Some example parameters for SEI growth from the papers:

        Ramadass, P., Haran, B., Gomadam, P. M., White, R., & Popov, B. N. (2004).
        Development of first principles capacity fade model for Li-ion cells. Journal of
        the Electrochemical Society, 151(2), A196-A203.

        Ploehn, H. J., Ramadass, P., & White, R. E. (2004). Solvent diffusion model for
        aging of lithium-ion battery cells. Journal of The Electrochemical Society,
        151(3), A456-A462.

        Single, F., Latz, A., & Horstmann, B. (2018). Identifying the mechanism of
        continued growth of the solid-electrolyte interphase. ChemSusChem, 11(12),
        1950-1955.

        Safari, M., Morcrette, M., Teyssot, A., & Delacour, C. (2009). Multimodal
        Physics- Based Aging Model for Life Prediction of Li-Ion Batteries. Journal of
        The Electrochemical Society, 156(3),

        Yang, X., Leng, Y., Zhang, G., Ge, S., Wang, C. (2017). Modeling of lithium
        plating induced aging of lithium-ion batteries: Transition from linear to
        nonlinear aging. Journal of Power Sources, 360, 28-40.

    Note: this parameter set does not claim to be representative of the true parameter
    values. Instead these are parameter values that were used to fit SEI models to
    observed experimental data in the referenced papers.

    SEI parameters
    ^^^^^^^^^^^^^^

    Parameters for lithium plating are from the paper:

        Yang, X., Leng, Y., Zhang, G., Ge, S., Wang, C. (2017). Modeling of lithium
        plating induced aging of lithium-ion batteries: Transition from linear to
        nonlinear aging. Journal of Power Sources, 360, 28-40.

    Note: this parameter set does not claim to be representative of the true parameter
    values. Instead these are parameter values that were used to fit SEI models to
    observed experimental data in the referenced papers.
    """

    return {
        "chemistry": "lithium_ion",
        # lithium plating
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
        "Lithium plating kinetic rate constant [m.s-1]": 1e-09,
        "Exchange-current density for plating [A.m-2]"
        "": plating_exchange_current_density_OKane2020,
        "Exchange-current density for stripping [A.m-2]"
        "": stripping_exchange_current_density_OKane2020,
        "Initial plated lithium concentration [mol.m-3]": 0.0,
        "Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Lithium plating transfer coefficient": 0.65,
        "Dead lithium decay constant [s-1]": 1e-06,
        "Dead lithium decay rate [s-1]": SEI_limited_dead_lithium_OKane2022,
        # sei
        "Ratio of lithium moles to SEI moles": 2.0,
        "Inner SEI reaction proportion": 0.5,
        "Inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "Outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        "SEI reaction exchange current density [A.m-2]": 1.5e-07,
        "SEI resistivity [Ohm.m]": 200000.0,
        "Outer SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        "Bulk solvent concentration [mol.m-3]": 2636.0,
        "Inner SEI open-circuit potential [V]": 0.1,
        "Outer SEI open-circuit potential [V]": 0.8,
        "Inner SEI electron conductivity [S.m-1]": 8.95e-14,
        "Inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        "Initial inner SEI thickness [m]": 2.5e-09,
        "Initial outer SEI thickness [m]": 2.55e-09,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity [m2.s-1]": 2e-18,
        "SEI kinetic rate constant [m.s-1]": 1e-12,
        "SEI open-circuit potential [V]": 0.4,
        "SEI growth activation energy [J.mol-1]": 38000.0,
        "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 0.5e-05,
        "Negative electrode thickness [m]": 60.6e-06,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 63.15e-06,
        "Positive current collector thickness [m]": 0.75e-05,
        "Electrode height [m]": 106e-3,
        "Electrode width [m]": 1.225,
        "Cell cooling surface area [m2]": 0.41,
        "Cell volume [m3]": 3.92e-05,
        "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        "Negative current collector conductivity [S.m-1]": 59600000.0,
        "Positive current collector conductivity [S.m-1]": 35500000.0,
        "Negative current collector density [kg.m-3]": 8954.0,
        "Positive current collector density [kg.m-3]": 2707.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 3.5,
        "Current function [A]": 3.5,
        "Contact resistance [Ohm]": 0,
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in negative electrode [mol.m-3]": 28746.0,
        "Negative electrode diffusivity [m2.s-1]": graphite_diffusivity_PeymanMPM,
        "Negative electrode OCP [V]": graphite_ocp_PeymanMPM,
        "Negative electrode porosity": 0.2,
        "Negative electrode active material volume fraction": 0.7545,
        "Negative particle radius [m]": 13.5E-06,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]": 1.2178795378580133e-06,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]": graphite_electrolyte_exchange_current_density_PeymanMPM,
        "Negative electrode density [kg.m-3]": 3100.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": graphite_entropic_change_PeymanMPM,
        "Negative electrode Poisson's ratio": 0.3,
        "Negative electrode Young's modulus [Pa]": 15000000000.0,
        "Negative electrode reference concentration for free of deformation [mol.m-3]": 0.0,
        "Negative electrode partial molar volume [m3.mol-1]": 3.1e-06,
        "Negative electrode volume change": graphite_volume_change_Ai2020,
        "Negative electrode initial crack length [m]": 2e-08,
        "Negative electrode initial crack width [m]": 1.5e-08,
        "Negative electrode number of cracks per unit area [m-2]": 3180000000000000.0,
        "Negative electrode Paris' law constant b": 1.12,
        "Negative electrode Paris' law constant m": 2.2,
        "Negative electrode cracking rate": graphite_cracking_rate_Ai2020,
        "Negative electrode LAM constant proportional term [s-1]": 2.7778e-07,
        "Negative electrode LAM constant exponential term": 2.0,
        "Negative electrode critical stress [Pa]": 60000000.0,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in positive electrode [mol.m-3]": 35380.0,
        "Positive electrode diffusivity [m2.s-1]": NMC_diffusivity_PeymanMPM,
        "Positive electrode OCP [V]": NMC_ocp_Siegel2022,
        "Positive electrode porosity": 0.3,
        "Positive electrode active material volume fraction": 0.6818,
        "Positive particle radius [m]": 2.1E-06,
        #Siegel2022,based on D50 targray specification of SNMC03015 / NMC622 (T61RS) Cathode material.
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
#        "Positive electrode transport efficiency": 0.16,
        "Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]"
        "": 1.3715539078341868e-07,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": NMC_electrolyte_exchange_current_density_PeymanMPM,
        "Positive electrode density [kg.m-3]": 3100.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1100.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]"
        "": NMC_entropic_change_PeymanMPM,
        "Positive electrode Poisson's ratio": 0.2,
        "Positive electrode Young's modulus [Pa]": 375000000000.0,
        "Positive electrode reference concentration for free of deformation [mol.m-3]"
        "": 0.0,
        "Positive electrode partial molar volume [m3.mol-1]": 1.25e-05,
        "Positive electrode volume change": volume_change_Ai2020,
        "Positive electrode initial crack length [m]": 2e-08,
        "Positive electrode initial crack width [m]": 1.5e-08,
        "Positive electrode number of cracks per unit area [m-2]": 3180000000000000.0,
        "Positive electrode Paris' law constant b": 1.12,
        "Positive electrode Paris' law constant m": 2.2,
        "Positive electrode cracking rate": cracking_rate_Ai2020,
        "Positive electrode LAM constant proportional term [s-1]": 2.7778e-07,
        "Positive electrode LAM constant exponential term": 2.0,
        "Positive electrode critical stress [Pa]": 375000000.0,
        # separator
        "Separator porosity": 0.3,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 516.67,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
#        "Separator transport efficiency ": 0.25,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.38,
        "Thermodynamic factor": 1.0,
        "Typical lithium ion diffusivity [m2.s-1]": 5.34e-10,
        "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Dix,
        "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Dix,
        # experiment
        "Reference temperature [K]": 298.15,
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
        "": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 5.0,
        "Total heat transfer coefficient [W.m-2.K-1]": 5.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.8,
        "Upper voltage cut-off [V]": 4.3,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 2.8,
        "Initial concentration in negative electrode [mol.m-3]": 48.8682,
        "Initial concentration in positive electrode [mol.m-3]": 31513.0,
        "Initial temperature [K]": 298.15,
        # citations
        "citations": ["OKane2022", "OKane2020", "Chen2020"],
        # "citation": "@book{Siegel2022, title={Jasons model reference manual}}",
    }
