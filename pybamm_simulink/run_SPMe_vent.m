clear; close all; 
%warning off; 
clc; 

% python file path setup 
p = py.sys.path;
p.insert(int32(0),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM') %path to pybamm package
p.insert(int32(1),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\venv\Lib\site-packages') % path to other libraries in venv
pybamm = py.importlib.import_module('pybamm');

% matlab file path setup 
addpath('C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink\casadi-windows-matlabR2016a-v3.5.5') % path to matlab casadi 


% change cell model params 
param = table2struct(readtable('sim_settings.csv'));
param.soc_init = 1;
param.soc_name = '100A';
param.k_capacity = 0.7;
param.k_R_tab = 1;
writetable(struct2table(param),'sim_settings.csv');

% Regenerate the casadi objects
py.pybamm_setup.main()

%% Model setup and sim
disp(string(datetime, 'HH:mm:SS')+'  Model set up');

% The simulink model name
mdl ='SPMe_PI_vent';

% Read in settings and calculate some derived parameters
ocv_init = 4.2;   % initial cell voltage
t_end = 60*5;
dt = param.dt; % needs to be the same as when generated pybamm casadi objects 
V_min = -3;
soc_min=0;
delta_sigma_max = 42.35277961;
% R = 8.3145; %J.mol-1.K-1
% A = 0.009; %m2 (0.121*0.074)
% DMC = [6.4338; 1413; -44.25]; %Antoine coeffs [A B C] 
% EC = [6.4897; 1836.57; -102.23];
% y_dmc = 0.7;
% y_ec = 1-y_dmc;         
% m_an = 0.0191; 
% M_C6 = 72/1000; %kg/mol
% x_sei_0 = 0.15;
% V_head_0 = 6.65e-06; %m-3
% L_p0 = 2.4e-3; %m for 2 poron
% E_p = 0.19e6; %Pa Young's modulus for poron
% % E_c = 1e12; %assume large by default
% alpha = 1.1e-6; %m/K
% P_atm = 101e3; %Pa
% P_crit = 158e3; %Pa
% % P0 = 14e3; %Pa
% R = 8.3145; %J.mol-1.K-1
% % k_b = 1.380649e-23; %m2.kg.s-2.K-1;
T_amb = param.T_amb; % K
% sigma_0 = 1.5120e+04;
p0 = [0.114319142120841  -0.663997904709863 ...
        1.686768366470837  -2.462409468590447   2.282769243949050 ...
        -1.401600551327618   0.576551904459066  -0.157044814050174 ...
        0.027351974716897  -0.002866912229499   0.000171009483610...
        0.000029999998915].*1e5;
param.ocv_init = polyval(p0,param.soc_init);   % initial cell voltage

% Run the simulations
Q_nom = 4.6;
Q_cell = Q_nom*param.k_capacity;
T_ref = 100+273.15;
V_ref = 1;
I_max= Q_nom*50;
tic;
N_sum = 1;
in(1:N_sum) = Simulink.SimulationInput(mdl); 
disp(string(datetime, 'HH:mm:SS')+'  Start simulink');
sol = sim(in, 'ShowProgress' ,true); % array
disp(string(datetime, 'HH:mm:SS')+'  Finish simulink');
t_eval = toc
disp(string(round(sol.tout(end)/t_eval, 2)) + ' sim to real time')

% Display error message
if ~isempty(sol.ErrorMessage)
    sol.ErrorMessage
end


%% plot results and compare to pybamm 
figure(1)
tiledlayout(3,2,'TileSpacing','tight', 'TileIndexing','columnmajor');

% plot signals
t = sol.tout;
I = squeeze(sol.I.Data);
soc = squeeze(sol.soc.Data);
V_measured = squeeze(sol.V_measured.Data);
V_cell = squeeze(sol.V_cell.Data);
T = squeeze(sol.T.Data);
delta_sigma = squeeze(sol.delta_sigma.Data);
ce = squeeze(sol.ce.Data);
cs_n = squeeze(sol.cs_n.Data);
cs_p = squeeze(sol.cs_p.Data);
x=linspace(0,1,size(ce,1));
r=linspace(0,1,size(cs_n,1));

ax1 = nexttile(1);
hold on 
yyaxis left
plot(t, I/Q_nom)
ylabel('Current [A/A.h]')
yyaxis right
plot(t, soc)
ylabel('SOC')
xlabel('Time [s]')
% soc(end)

ax2 = nexttile(2);
hold on 
plot(t, V_measured,DisplayName='Measured voltage')
% plot(t, V_cell,DisplayName='Cell voltage')
ylabel('Measured voltage [V]')
xlabel('Time [s]')
ylim([-0.5,4])

ax3 = nexttile(3);
hold on 
plot(t, T-273.15)
plot(t([1, end]), ones(2,1)*(T_ref-273.15), '--k')
text(t(end)/2,T_ref-273.15,'Reference', Color='k', VerticalAlignment='bottom', HorizontalAlignment='center')
ylabel('Temperature [degC]')
xlabel('Time [s]')
ylim([T_amb-273.15, 120])

% ax4=nexttile;
% plot(r, cs_n(end,length(cs_n)))
% ylabel('$c_\mathrm{s,n}$')
% xlabel('r')
% legend(string(t(length(ce))) + 's', NumColumns=3, Location="ne")

ax4=nexttile;
hold on 
plot(t, cs_n(end,:))
% plot(t([1, end]), ones(2,1)*(0), '-k', linewidth=0.1)
ylabel('$c_\mathrm{se,n}$')
xlabel('Time [s]')


% ax5=nexttile;
% plot(r, cs_p(:,length(cs_p)))
% ylabel('$c_\mathrm{s,p}$')
% xlabel('r')
% legend(string(t(length(ce))) + 's', NumColumns=3, Location="ne")

ax5=nexttile;
plot(t, ce(end,:))
ylabel('$c_\mathrm{e}$')
xlabel('Time [s]')

ax6=nexttile;
hold on
plot(t, delta_sigma/1000)
plot(t([1, end]), ones(2,1)*(delta_sigma_max), '--r')
text(t(end)/2,delta_sigma_max,'Venting', Color='r', VerticalAlignment='bottom', HorizontalAlignment='center')
ylabel('$\Delta\sigma$ [kPa]')
xlabel('Time [s]')

% format plot
% linkaxes([ax1,ax2,ax3],"x")
nexttile(1)
% legend(NumColumns=2, Location="se")
set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',10)
set(gcf,'Position',[40 60 900,450 ])
