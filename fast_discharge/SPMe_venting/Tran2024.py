import pybamm
from pybamm import exp, constants, Parameter
import numpy as np


def graphite_ocp_PeymanMPM(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].

    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)
    """

    u_eq = (
        0.063
        + 0.8 * np.exp(-75 * (sto + 0.007))
        - 0.0120 * np.tanh((sto - 0.127) / 0.016)
        - 0.0118 * np.tanh((sto - 0.155) / 0.016)
        - 0.0035 * np.tanh((sto - 0.220) / 0.020)
        - 0.0095 * np.tanh((sto - 0.190) / 0.013)
        - 0.0145 * np.tanh((sto - 0.490) / 0.020)
        - 0.0800 * np.tanh((sto - 1.030) / 0.055)
    )

    return u_eq



def NMC_ocp_PeymanMPM(sto):
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
        4.3452
        - 1.6518 * sto
        + 1.6225 * (sto**2)
        - 2.0843 * (sto**3)
        + 3.5146 * (sto**4)
        - 2.2166 * (sto**5)
        - 0.5623 * np.exp(109.451 * sto - 100.006) #e-4
    )

    return u_eq

def modified_graphite_diffusivity_PeymanMPM(sto, T):
    D_ref =  Parameter("Negative electrode diffusion coefficient [m2.s-1]")
    # D_ref = 16*5.0 * 10 ** (-15)
    # E_D_s = 42770
    # arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))
    soc = (sto - 0)/(0.8321-0)
    k = 1.12070451*soc + 0.09209274 # Ds_restart_rmseV_11_simultaneous_rest (updated exp C-rate) only exclude kn>9

    # soc_fit = np.array([0.931499472,0.862995346,0.79449311,0.725995244,0.657493658,0.588993969,0.520494998,0.246491535,0.177990605,0.109487916][-1:0:-1])
    # k_fit = np.array([0.998960993,0.962562613,1.088848875,0.960830764,0.980577779,0.767837023,0.826562007,0.192714141,0.571586523,0.725658028][-1:0:-1])
    # x = [soc]
    # k = pybamm.Interpolant(soc_fit, k_fit, x, name=None, interpolator='linear', extrapolate=True, entries_string=None)
    # k = pybamm.maximum(k,1)
    return D_ref*k #*arrhenius # *(-0.9 * sto + 1)

def modified_NMC_diffusivity_PeymanMPM(sto, T):
    D_ref =  Parameter("Positive electrode diffusion coefficient [m2.s-1]")
    # E_D_s = 18550
    # arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))
    soc = (0.837-sto)/(0.837-0.034)
    k =  4.7302281*soc**2  -4.5023245*soc + 1.26466141 # Ds_restart_rmseV_11_simultaneous_rest (updated exp C-rate)
    # soc_fit = np.array([0.931499472,0.862995346,0.79449311,0.725995244,0.657493658,0.588993969,0.520494998,0.451992894,0.383496394,0.31499461,0.246491535,0.109487916][-1:0:-1])
    # k_fit = np.array([1.224990159,0.918644845,0.646317414,0.418199795,0.250350793,0.318672366,0.267289649,0.182154457,0.11139899,0.329794626,0.337200374,0.418978523][-1:0:-1])
    # x = [soc]
    # k = pybamm.Interpolant(soc_fit, k_fit, x, name=None, interpolator='linear', extrapolate=True, entries_string=None)
    # k = pybamm.maximum(k,1)
    return D_ref *k #*arrhenius

def modified_electrolyte_diffusivity_PeymanMPM(c_e, T):
    # D_c_e = 5.35 * 10 ** (-10)
    D_c_e =  Parameter("Typical electrolyte diffusivity [m2.s-1]")
    return D_c_e

def modified_electrolyte_conductivity_PeymanMPM(c_e, T):
    # sigma_e = 1.3
    sigma_e = Parameter("Typical electrolyte conductivity [S.m-1]")
    E_k_e = 34700
    return sigma_e


def modified_NMC_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    m_ref =  Parameter("Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]")
    # m_ref = 4.824 * 10 ** (-6)  # (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 39570
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 /T))
    return (
        m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5 #* arrhenius
    )

def modified_graphite_electrolyte_exchange_current_density_PeymanMPM(c_e, c_s_surf, c_s_max, T):
    m_ref =  Parameter("Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]")
    # m_ref = 4*1.061 * 10 ** (-6)  # unit has been converted units are (A/m2)(mol/m3)**1.5 - includes ref concentrations
    E_r = 37480
    arrhenius = exp(E_r / constants.R * (1 / 298.15 - 1 / T))

    return (
        m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5 #*arrhenius
    )

def P_sat_Tran2024(T):
    """
    Electrolyte saturation pressure as a function of temperature [K]

    References
    ----------
    .. [1] 

    Parameters
    ----------
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Electrolyte saturation pressure [kPa]
    """
    P_dmc = 10**(6.4338-1413.0/(T-44.25))
    P_ec = 10**(6.4897-1836.57/(T-102.23))
    P = P_ec*0.3 + P_dmc*0.7 
    return P

Cps = {'100A': 2.4847642158614396, '100B': 2.131841251256563, '75': 1.7615851440073333, '50': 1.8069119898506183}    # 12/31 test   T only, no weights
hs = {'100A': 34.83874617168782, '100B': 34.211742173359575, '75': 37.17732764082647, '50': 35.31863975361435}   
R_tabs = {'100A':  0.0086, '100B':0.0049, '75':0.0078+0.0005, '50':0.007-0.0007}
R_escs = {'100A':  0.0067, '100B':0.004, '75':0.0067-0.0005, '50':0.0067+0.0007}
k_A_sei  = {'100A': 0.13210066, '100B': 0.320855, '75': 0.06343707, '50':0.08423777}# k_A = 0.05716926 (MSE 1e-8 50%), k_A = 0.04571944 (MSE 1e-8 75%), k_A = 0.07912785(MSE 1e-8 100% t<80s), k_A = 0.238155 (MSE 1e-8 100F% t<53s)

# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for a graphite/NMC532 pouch cell from the paper :footcite:t:`Tran2024`
    and references therein.

    SEI parameters are example parameters for SEI growth from the papers
    :footcite:t:`Ramadass2004`, :footcite:t:`ploehn2004solvent`,
    :footcite:t:`single2018identifying`, :footcite:t:`safari2008multimodal`, and
    :footcite:t:`Yang2017`

    SEI parameters
    ^^^^^^^^^^^^^^

    Parameters for lithium plating are from the paper :footcite:t:`Yang2017`

    .. note::
        This parameter set does not claim to be representative of the true parameter
        values. Instead these are parameter values that were used to fit SEI models to
        observed experimental data in the referenced papers.
    """

    return {
        "chemistry": "lithium_ion",
        # lithium plating
        "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
        "Exchange-current density for plating [A.m-2]": 0.001,
        "Initial plated lithium concentration [mol.m-3]": 0.0,
        "Typical plated lithium concentration [mol.m-3]": 1000.0,
        "Lithium plating transfer coefficient": 0.7,
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
        "Initial outer SEI thickness [m]": 2.5e-09,
        "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        "EC diffusivity [m2.s-1]": 2e-18,
        "SEI kinetic rate constant [m.s-1]": 1e-12,
        "SEI open-circuit potential [V]": 0.4,
        "SEI growth activation energy [J.mol-1]": 0.0,
        "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # cell
        "Negative current collector thickness [m]": 2.5e-05,
        "Negative electrode thickness [m]": 6.2e-05*4.2/5,
        "Separator thickness [m]": 1.2e-05,
        "Positive electrode thickness [m]": 6.7e-05*4.2/5,
        "Positive current collector thickness [m]": 2.5e-05,
        "Electrode height [m]": 1.0,
        "Electrode width [m]": 0.205,
        "Cell cooling surface area [m2]": 0.025549 * (3.49238e-05/6.3635E-5),#calculated Real dim ~[130mm×89mm×5.5mm], scaled to account for volume change between pybamm 2022 and 2024  
        "Cell volume [m3]": 3.49238e-05, #3.92e-05, #6.3635E-5, #L*L_y*L_z (L includes ccs.  L_x doesn't)
        "Negative current collector conductivity [S.m-1]": 5.96e7,
        "Positive current collector conductivity [S.m-1]": 3.55e7,
        "Negative current collector density [kg.m-3]": 8954.0,
        "Positive current collector density [kg.m-3]": 2707.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 4.6,
        "Current function [A]": 4.6,
        "Contact resistance [Ohm]": 0,
        "Tabbing resistance [Ohm]": 0, 
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in negative electrode [mol.m-3]": 28746.0,
        "Negative particle diffusivity [m2.s-1]": modified_graphite_diffusivity_PeymanMPM,
        "Negative electrode OCP [V]": graphite_ocp_PeymanMPM,
        "Negative electrode porosity": 0.3,
        "Negative electrode active material volume fraction": 0.61,
        "Negative particle radius [m]": 10E-06, #2.5e-06,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode transport efficiency": 0.16,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]": modified_graphite_electrolyte_exchange_current_density_PeymanMPM,
        "Negative electrode density [kg.m-3]": 3100.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100.0*Cps['100A'],
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": 0, # graphite_entropic_change_PeymanMPM,
        "Negative electrode reference exchange-current density [A.m-2(m3.mol)1.5]":4.244E-6, #1.061e-06,
        "Negative electrode diffusion coefficient [m2.s-1]": 8.0E-14,
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 100.0,
        "Maximum concentration in positive electrode [mol.m-3]": 35380.0,
        "Positive particle diffusivity [m2.s-1]": modified_NMC_diffusivity_PeymanMPM,
        "Positive electrode OCP [V]": NMC_ocp_PeymanMPM,
        "Positive electrode porosity": 0.3,
        "Positive electrode active material volume fraction": 0.445,
        "Positive particle radius [m]": 3.5e-06,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode transport efficiency": 0.16,
        "Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]": 4.824e-06,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]": modified_NMC_electrolyte_exchange_current_density_PeymanMPM,
        "Positive electrode density [kg.m-3]": 3100.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1100.0*Cps['100A'],
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": 0,#NMC_entropic_change_PeymanMPM,
        "Positive electrode diffusion coefficient [m2.s-1]":8.0E-15, # added
        "Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5]": 4.824E-06,
        # separator
        "Separator porosity": 0.4,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        "Separator transport efficiency ": 0.25,
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Cation transference number": 0.38,
        "Thermodynamic factor": 1.0,
        "Typical lithium ion diffusivity [m2.s-1]": 5.35e-10,
        "Electrolyte diffusivity [m2.s-1]": modified_electrolyte_diffusivity_PeymanMPM,
        "Electrolyte conductivity [S.m-1]": modified_electrolyte_conductivity_PeymanMPM,
        "Typical electrolyte diffusivity [m2.s-1]": 5.35E-10,
        "Typical electrolyte conductivity [S.m-1]":1.3,
        # experiment
        "Reference temperature [K]": 298.15,
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0.0,
        "Total heat transfer coefficient [W.m-2.K-1]": hs['100A'],
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 0,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 2.8,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 48.8682,
        "Initial concentration in positive electrode [mol.m-3]": 31513.0,
        "Initial temperature [K]": 298.15,
        # "External heating [W.m-3]": 0,
        # venting 
        "Active material surface area [m2]": 0.009,
        "Initial head space volume [m3]": 6.65e-06,
        "Poron sheet thickness [m]":0.0024, #thickness of two sheets, 
        "Initial cell compression stress [kPa]": 1.511989742044445e+01,
        "Young's modulus of the poron sheet [kPa]": 190, 
        "Atmospheric pressure [kPa]": 101.325,
        "Thermal expansion coefficient of the cell [m.K-1]": 1.1e-6,
        "Critical venting pressure [kPa]": 1.587731e02,
        # electrolyte vaporization  (unused except P_sat)
        "Electrolyte saturation pressure [kPa]":P_sat_Tran2024,
        "Molar mass of electrolyte [kg.mol-1]": 0.099295, # EC:EMC 3:7 88.06*.3+104.11*0.7,
        "Density of electrolyte [kg.m-3]": 1100, #from Cell Press electrolyte costs supplemental material,
        "Initial amount of electrolyte [kg]": 0.020, #Assume 4g electrolyte for every Ah
        # positive electrode decomposition
        "Frequency factor for cathode decomposition [s-1]":2.55E14, 
        "Activation energy for cathode decomposition [J]":2.64E-19,
        "Enthalpy of cathode decomposition [J.kg-1]":790000, 
        "Initial degree of conversion of cathode decomposition": 0.04,
        # negative electrode decomposition
        "Frequency factor for anode decomposition [s-1]": 2.5E13, # Cai 2019,
        "Activation energy for anode decomposition [J]":2.24E-19, #Cai 2019,
        "Enthalpy of anode decomposition [J.kg-1]":1714000, #Cai 2019,
        # SEI decomposition
        "Frequency factor for SEI decomposition [s-1]": 1.67E15*k_A_sei['100A'], #Coman 2017 (2.25E15 in Cai 2019 is wrong),
        "Activation energy for SEI decomposition [J]":2.24E-19,# Cai 2019,
        "Enthalpy of SEI decomposition [J.kg-1]":257000, #Cai 2019,
        "Initial fraction of Li in SEI": 0.15, #Cai 2019,
        "Initial SEI thickness": 0.033, #Cai 2019,
        "Mass of the negative electrode active material [kg]": 0.019107, #Cai 2019,
        "Molar mass of negative electrode active material [kg.mol-1]": 0.072,# For C6,
        # citations
        "citations": ["Tran2024"],
    }
