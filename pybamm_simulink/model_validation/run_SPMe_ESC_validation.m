% clear; close all; 
%warning off; 
clc; 

% set python filepaths
p = py.sys.path;
p.insert(int32(0),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM') %path to pybamm package
p.insert(int32(1),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\venv\Lib\site-packages') % path to other libraries in venv
pybamm = py.importlib.import_module('pybamm');

% set matlab filepaths
addpath('C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink')  
addpath('C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink\casadi-windows-matlabR2016a-v3.5.5')  
disp([datestr(now, 'HH:MM:SS'),'  Model set up']);

param_file = 'sim_settings.csv';

% The simulink model name
mdl ='SPMe_ESC_validation_simulink_external_T'; %'pack_2s1p';% 'pybamm_SPMe'; %

% Regenerate the casadi objects
py.pybamm_setup_validation_external_T.main(); %_external_T.main()

% Read in settings and calculate some derived parameters
param = table2struct(readtable(param_file));
T_amb = param.T_amb; %25
V_min = 0;
R_tab = 0.0086; %{'100A':  0.0086, '100B':0.0049, '75':0.0078+0.0005, '50':0.007}
R_esc = 0.0067; %{'100A':  0.0067, '100B':0.004, '75':0.0067-0.0005, '50':0.0067}
t_end = 60*1;

% Load venting parameters and T_amb
% load('venting_params.mat')

% thermal model params (internal)
% A_cool = 0.025549; % Cell cooling surface area [m2] pouch 0.41 default Real dim ~[130mm×89mm×5.5mm] 
% V_cell = 6.3635E-5; % Cell volume [m3],pouch 3.92E-5 Real dim ~[130mm×89mm×5.5mm] 
% % rho_Cp_eff = 6271195.03973258; 
% Cp = 2733; %rho_Cp_eff/ (1100*2.4847642158614396); % hack. Cp from electrode, not cell
% m_cell = 0.13;
% h_eff = 34.8;
A_cool = param.A_cool; % Cell cooling surface area [m2] pouch 0.41 default Real dim ~[130mm×89mm×5.5mm] 
mCp = param.mCp; % hack. Cp from electrode, not cell
m_cell = 0.13; % kg
Cp = mCp/m_cell;
h_eff = param.h_total;
V_cell = param.V_cell;

% Run the simulation
tic;
N_sum = 1;
param.endtime = 5*60;
% param.ocv_init = 4.2; % for mdl with power system  blocks
in(1:N_sum) = Simulink.SimulationInput(mdl);
disp([datestr(now, 'HH:MM:SS'),'  Start simulink']);
sol = sim(in, 'ShowProgress' ,true);
disp([datestr(now, 'HH:MM:SS'),'  Finish simulink']);
toc;

% Display error message
if ~isempty(sol.ErrorMessage)
    sol.ErrorMessage
end

%% plot results and compare to pybamm 
pybamm_sol = readtable('.\pybamm_ESC_sim_results\pybamm_100A_ESC_sim_results.csv');

figure(1)
t=sol.tout;
I = squeeze(sol.I.Data);
V = squeeze(sol.V.Data);
T = squeeze(sol.T.Data);
Q = squeeze(sol.Q.Data);
ce = squeeze(sol.ce.Data);
cs_n = squeeze(sol.cs_n.Data);
cs_p = squeeze(sol.cs_p.Data);
x=linspace(0,1,size(ce,1));
r=linspace(0,1,size(cs_n,1));

tiledlayout(3,2,'TileSpacing','tight', 'TileIndexing','columnmajor');
nexttile;
hold on 
plot(pybamm_sol.t, pybamm_sol.I, '-k', DisplayName='PyBaMM')
plot(t, I, '--r', DisplayName='Simulink')
ylabel('Current [A]')
xlabel('Time [s]')
xlim([0,t(end)])
ylim([0, 250])
legend

nexttile;
hold on 
plot(pybamm_sol.t, pybamm_sol.V, '-k')
plot(t, V, '--r')
ylabel('Measured voltage [V]')
xlabel('Time [s]')
xlim([0,t(end)])
ylim([0,2.5])

nexttile;
hold on 
plot(pybamm_sol.t, pybamm_sol.T-273.15, '-k')
plot(t, T-273.15, '--r')
xlim([0,t(end)])
ylabel('Temperature [degC]')
xlabel('Time [s]')

nexttile;
% plot(x, ce(:,length(ce)))
% ylabel('$c_\mathrm{e}$')
% legend(string(t(length(ce))) + 's', NumColumns=3, Location="ne")
% xlabel('x')
plot(t, Q,  '--r')
xlim([0,t(end)])
ylim([0,5e7])
ylabel('Q [W.m-3]')

nexttile;
plot(r, cs_n(:,length(cs_n)))
ylabel('$c_\mathrm{s,n}$')
xlabel('r')
legend(string(t(length(ce))) + 's', NumColumns=3, Location="ne")


nexttile;
plot(r, cs_p(:,length(cs_p)))
ylabel('$c_\mathrm{s,p}$')
xlabel('r')

set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',12)
set(gcf,'Position',[40 60 900,900])
