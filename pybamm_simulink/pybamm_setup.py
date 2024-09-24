# -*- coding: utf-8 -*-
"""
Created on 06/17/2024

@author: vivian
"""

import sys
import os
# print(os.getcwd())
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
# print(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
# os.chdir(os.path.dirname(os.path.abspath("__file__")))
# import pybamm
# print(pybamm.__path__[0])
# from pybamm import exp, constants, Parameter
# #import matplotlib.pyplot as plt
# move back to folder containing this file
# os.chdir(sys.path[0])

# for debugging
# sys.path.append(os.path.dirname(os.path.abspath("__file__"))) # for debugging
# os.chdir(sys.path[0])


import pybamm
print(pybamm.__path__[0])
from pybamm import exp, constants, Parameter
import numpy as np
from scipy import io
import os
import shutil
import casadi
import pandas as pd
import pickle
from scipy.interpolate import interp1d

class ExternalCircuitResistanceFunction():
    def __call__(self, variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        R_ext = pybamm.FunctionParameter("External short resistance [Ohm]",  {"Time [s]": pybamm.t}) 
        R_tab = pybamm.FunctionParameter("Tabbing resistance [Ohm]",  {"Time [s]": pybamm.t})
        return V/I - (R_ext + R_tab)

def modified_graphite_diffusivity_PeymanMPM(sto, T):
    D_ref =  Parameter("Negative electrode diffusion coefficient [m2.s-1]")
    # D_ref = 16*5.0 * 10 ** (-15)
    # E_D_s = 42770
    # arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))
    soc = (sto - 0)/(0.8321-0)
    k = 1.12070451*soc + 0.09209274 # ORIGINAL!! Ds_restart_rmseV_11_simultaneous_rest (updated exp C-rate) only exclude kn>9
    # k = 47.43212855*soc**4 + -119.85420669*soc**3 + 104.9741128*soc**2  + -35.54312648*soc  +  4.18552321
    k = pybamm.maximum(k, 0.1)
    # soc_fit = np.array([0.931499472,0.862995346,0.79449311,0.725995244,0.657493658,0.588993969,0.520494998,0.246491535,0.177990605,0.109487916][-1:0:-1])
    # k_fit = np.array([0.998960993,0.962562613,1.088848875,0.960830764,0.980577779,0.767837023,0.826562007,0.192714141,0.571586523,0.725658028][-1:0:-1])
    # x = [soc]
    # k = pybamm.Interpolant(soc_fit, k_fit, x, name=None, interpolator='linear', extrapolate=True, entries_string=None)
    
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
    
    return D_ref #*k #*arrhenius

def modified_electrolyte_diffusivity_PeymanMPM(c_e, T):
    # D_c_e = 5.35 * 10 ** (-10)
    D_c_e =  Parameter("Typical electrolyte diffusivity [m2.s-1]")
    return D_c_e

def modified_electrolyte_conductivity_PeymanMPM(c_e, T):
    # sigma_e = 1.3
    sigma_e = Parameter("Typical electrolyte conductivity [m2.s-1]")
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

def modified_graphite_ocp(sto):
    """
    Graphite Open Circuit Potential (OCP) as a function of the
    stochiometry. The fit is taken from Peyman MPM [1].

    References
    ----------
    .. [1] Peyman Mohtat et al, MPM (to be submitted)
    """

    u_eq = (
        0.063
        + 0.8 * pybamm.exp(-75 * (sto + 0.007))
        # + 0.8 * exp(-100 * (sto + 0.00))
        - 0.0120 * pybamm.tanh((sto - 0.127) / 0.016)
        - 0.0118 * pybamm.tanh((sto - 0.155) / 0.016)
        - 0.0035 * pybamm.tanh((sto - 0.220) / 0.020)
        - 0.0095 * pybamm.tanh((sto - 0.190) / 0.013)
        - 0.0145 * pybamm.tanh((sto - 0.490) / 0.020)
        - 0.0800 * pybamm.tanh((sto - 1.030) / 0.055)
    )

    return u_eq

def main(plot=False):
    pybamm.settings.max_smoothing = 'exact' # 'exact'(default), 10 (recommended)
    settings = pd.read_csv('sim_settings.csv').to_dict(orient='index')[0]
    dt = settings['dt']
    soc_init = settings['soc_init']
    Q_nom = 4.6 #Ah

    # create model
    R_tab = pybamm.Parameter("Tabbing resistance [Ohm]")
    R_ext = pybamm.Parameter("External resistance [Ohm]")
    model_options = {
        "thermal": "lumped",
        # "external submodels": ["thermal"],
        "decomposition": "true", 
        # "cell geometry": "arbitrary",
        # "operating mode": ExternalCircuitResistanceFunction(),
        # "venting":"true",
    }
    model = pybamm.lithium_ion.SPMe(model_options)

    # setup parameters
    chemistry = pybamm.parameter_sets.Tran2023
    param = pybamm.ParameterValues(chemistry=chemistry)
    SOC_name = str(settings['soc_name'])
    Cps = {'100A': 2.4847642158614396, '100B': 2.131841251256563, '75': 1.7615851440073333, '50': 1.8069119898506183}    # 12/31 test   T only, no weights
    hs = {'100A': 34.83874617168782, '100B': 34.211742173359575, '75': 37.17732764082647, '50': 35.31863975361435}   
    R_tabs = {'100A':  0.0086, '100B':0.0049, '75':0.0078+0.0005, '50':0.007}
    R_escs = {'100A':  0.0067, '100B':0.004, '75':0.0067-0.0005, '50':0.0067}
    # k_A_sei = {'100A':  0.07912785, '100B':0.238155, '75':0.04571944, '50':0.05716926}
    k_A_sei = {'100A': 0.13210066, '100B': 0.320855, '75': 0.06343707, '50':0.08423777}# k_A = 0.05716926 (MSE 1e-8 50%), k_A = 0.04571944 (MSE 1e-8 75%), k_A = 0.07912785(MSE 1e-8 100% t<80s), k_A = 0.238155 (MSE 1e-8 100F% t<53s)
    T_amb = 25
    sigma_0 = 15
    k_capacity = settings['k_capacity']
    k_R_tab = settings['k_R_tab']
    param.update({
        "Current function [A]": "[input]",
        "Frequency factor for SEI decomposition [s-1]": 1.67E15*k_A_sei[SOC_name],
        "Tabbing resistance [Ohm]":  R_tabs[SOC_name]*k_R_tab,#0.0041,D
        "External short resistance [Ohm]": R_escs[SOC_name], # 0.0067
        "Negative electrode thickness [m]":62E-6*4.2/5,
        "Positive electrode thickness [m]":67E-6*4.2/5,
        "Negative electrode porosity": 0.3*k_capacity,
        "Positive electrode porosity": 0.3*k_capacity,
        "Lower voltage cut-off [V]": 0,
        "Ambient temperature [K]":T_amb + 273.15,
        "Initial temperature [K]": T_amb + 273.15,
        "Initial cell compression stress [kPa]": sigma_0, #data['F'].iloc[0]/param['Active material surface area [m2]']/1000,
        "Total heat transfer coefficient [W.m-2.K-1]":hs[SOC_name],
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 1100*Cps[SOC_name],
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1100*Cps[SOC_name],
        "Negative electrode diffusivity [m2.s-1]": modified_graphite_diffusivity_PeymanMPM,
        "Positive electrode diffusivity [m2.s-1]": modified_NMC_diffusivity_PeymanMPM,
        "Electrolyte diffusivity [m2.s-1]": modified_electrolyte_diffusivity_PeymanMPM,
        "Electrolyte conductivity [S.m-1]": modified_electrolyte_conductivity_PeymanMPM,
        # "Initial concentration in electrolyte [mol.m-3]":1000, #*1.3
        "Negative electrode exchange-current density [A.m-2]": modified_graphite_electrolyte_exchange_current_density_PeymanMPM,
        "Positive electrode exchange-current density [A.m-2]": modified_NMC_electrolyte_exchange_current_density_PeymanMPM,
        "Negative electrode OCP entropic change [V.K-1]":0,
        "Positive electrode OCP entropic change [V.K-1]":0,
        "Electrode height [m]": 1.0,
        "Electrode width [m]": 0.205*Q_nom/4.6,
        # "Negative electrode OCP [V]": modified_graphite_ocp,
    }, check_already_exists = False)
    # liion = pybamm.LithiumIonParameters()
    # rho = param.evaluate(liion.therm.rho_eff_dim(T_amb+273.15)) #eff vol heat cap
    # cell_surface_area = param.evaluate(liion.A_cooling)
    # cell_volume = param.evaluate(liion.V_cell)
    # h_total = param.evaluate(liion.therm.h_total_dim)
    # print(rho)
    # print(cell_surface_area)
    # print(cell_volume)
    # print(h_total)
    # print(total_cooling_coefficient)

    # update voltage definition 

    V = model.variables["Terminal voltage [V]"]
    I = model.variables["Current [A]"]
    model.variables.update({
        "Measured voltage [V]": V - I*R_tabs['100A'],
        "Actual resistance [Ohm]":V/I,
        }
    )
    
    # Solve for very short time - we're just initialising here
    solver = pybamm.CasadiSolver(mode='safe')
    sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
    inputs = {
        "Current function [A]":Q_nom,
        }
    t_eval = np.linspace(0, 1e-6, 3)
    sim.solve(t_eval=t_eval, inputs=inputs, initial_soc=soc_init)
              
    # Save the inital states
    x0 = sim.solution.y.full()[:, 0]
    cwd = os.getcwd()
    temp_dir = os.path.join(cwd, 'temp')
    # temp_dir = os.path.join(cwd, 'temp_' + str(int(dt*1000))+'ms')

    shutil.rmtree(temp_dir, ignore_errors=True)
    os.mkdir(temp_dir)
    io.savemat(os.path.join(temp_dir, 'x0.mat'), {'x0':x0})
    
    # Create integrator for specified time interval
    t_eval = np.linspace(0, dt, 50)
    t_eval_ndim = t_eval / sim.built_model.timescale.evaluate(inputs=inputs)
    inp_and_ext = inputs
    # inp_and_ext.update(external_variables)
    casadi_integrator = solver.create_integrator(sim.built_model, inputs=inp_and_ext, t_eval=t_eval_ndim)
    # Save the integrator and variables function
    ci_path = os.path.join(temp_dir, 'integrator.casadi')
    casadi_integrator.save(ci_path)
    # These will be the outputs from the simulink class so must be changed there too
    variable_names = ['Terminal voltage [V]',
                      'Measured battery open circuit voltage [V]',
                      'Volume-averaged cell temperature [K]',
                      'Electrolyte concentration', #'Cell expansion stress [kPa]',
                      'X-averaged total heating [W.m-3]',
                      'X-averaged negative particle concentration', # [mol.m-3]
                      'X-averaged positive particle concentration',
                      'Fraction of Li in SEI',
                      'Measured voltage [V]',
                      'Negative electrode open circuit potential [V]',
                      'Positive electrode open circuit potential [V]',
                      ]
    ipo = ["Current function [A]"] # input parameter order
    casadi_objs = sim.built_model.export_casadi_objects(variable_names=variable_names,
                                                        input_parameter_order=ipo)
    variables = casadi_objs['variables']
    t, x, z, p = casadi_objs["t"], casadi_objs["x"], casadi_objs["z"], casadi_objs["inputs"]
    variables_stacked = casadi.vertcat(*variables.values())
    variables_fn = casadi.Function("variables", [t, x, z, p], [variables_stacked])
    v_path = os.path.join(temp_dir, 'variables.casadi')
    variables_fn.save(v_path)
    print('Casadi objects re-generated \n', os.listdir(temp_dir))
    print('When changing python script please restart Matlab for changes to take effect')
    
    if plot:
        # If testing run the simulation for longer and plot
        sim.solve(t_eval=np.linspace(0, 60*8, 60*8*10), inputs=inputs)

        # plot
        p = pybamm.QuickPlot(sim.solution, output_variables=variable_names)
        p.dynamic_plot()
    
    # print(casadi.__version__)

    # x = sim.solution["x [m]"].entries[:, 0]
    # t = sim.solution["Time [s]"].entries
    # c_e= sim.solution['Electrolyte concentration']
    # c_s_p_surf = sim.solution['Positive particle surface concentration']
    # c_s_n_surf = sim.solution['Negative particle surface concentration']


if __name__ == '__main__':
    main(plot=False)
    
