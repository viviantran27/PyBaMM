% I created this variant to run the pybamm model to test PI control on temp
% ************* Run section by section!*************
clear; close all; 
clc; 

% casadi path setup
p = py.sys.path;
p.insert(int32(0),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\venv\Lib\site-packages') % path to python casadi
addpath('C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink') % path to matlab casadi 
casadi_solver = py.importlib.import_module('casadi'); % import python casadi

%% Model setup and sim
disp(string(datetime, 'HH:mm:SS')+'  Model set up');
% Read in settings and calculate some derived parameters
ocv_init    = 4.2;   % initial cell voltage
t_end       = 5000; 
dt          = 0.05; % needs to be the same as when generated pybamm casadi objects 
V_min       = 0;

tic;
Q_nom = 4.6;
N_sum = 1;
mdl = 'Bat_SPM';
in(1:N_sum) = Simulink.SimulationInput(mdl); 
disp(string(datetime, 'HH:mm:SS')+'  Start simulink');
toc;

Q           = 4.6*3600;         % [Ah to As]
Rs          = [0.051109270053841  -0.124390708105661...
            0.112323360710586  -0.046901650715843 0.015533047683140]...
            *0.5.^(4:-1:0)';
Rbat1       = [0.030297671277692  -0.078249784854926...
            0.076106125220171  -0.035096138367922   0.013959006299232]...
            *0.5.^(4:-1:0)';
Cbat1       = [-0.178332712122388   0.724802617995660 ...
            -1.060271845888578   0.759649001977427   0.210118417436246]...
            *1e4*0.5.^(4:-1:0)';

alpha       = 64.53;            % [-]
beta        = 0.48;             % [-]
% alpha       = 1;            % [-]
% beta        = 1;             % [-]
Tamb        = 293.8;            % [K]
h           = 54.06;            % [W/K.m^2]
A_surf      = 0.009;            % [m^2]
kb          = 1.380649*1e-23;   % [J/K]   
E_SEI       = 2.24*1e-19;       % [J]
A_SEI       = 2.25*1e15;        % [s^-1]
m_cell      = 104/1000;         % [kg]
Cp          = 2108;             % [J/kg.K]
m_an        = 19.107;           % [g]
h_SEI       = 257;              % [J/g^-1]
Parameters = [Q,Rs,Rbat1,Cbat1,alpha,beta,Tamb,h,A_surf,kb,E_SEI,A_SEI,m_cell,Cp,m_an,h_SEI];

% Controller parameters
Imax        = 46;               % [A] (10C)
Tmax        = 80+273.15;        % [K]

CoefP       = [2.343336364187066e-09, -2.176092796402463e-06, 6.938605783813887e-04, 0.071489987134737];    % Polynomial gains
K           = [1,1e-1,3.5e-2];                                  % [SoC, T, Pressure]

Controller_Params = [Imax,Tmax,Q,Rs,h,A_surf,Tamb,m_cell,Cp,A_SEI,E_SEI,kb,CoefP,K];

% Pressure model parameters
P_crit      = 158e3;            % [Pa]
P0          = 4.1374e3;         % [Pa]

xSEI0       = 0.15;             % [-]
MC6         = 72.06;            % [g/mol]
R           = 8.314;            % [J/K⋅mol]

P_crit      = 158e3;            % [Pa]
P_atm       = 101e3;            % [Pa]
alpha_cell  = 1.1e-6;           % [m/K]
E           = 0.19e6;           % [Pa]
L           = 2.4e-3;           % [m]
Vh0         = 6.65*1e-6;        % [m^3]
sig0        = 1.5120e7;         % [Pa]
A1          = 6.4897;           % EC
B1          = 1836.57;          % EC
C1          = -102.23;          % EC

A2          = 6.4338;           % DMC
B2          = 1413.0;           % DMC
C2          = -44.25;           % DMC

yEC         = 0.3;              % EC
yDMC        = 0.7;              % DMC

Params_P = [sig0,P_atm,E,alpha_cell,Tamb,L,Vh0,A_surf,m_an,xSEI0,MC6,R,A1,B1,...
    C1,A2,B2,C2,yEC,yDMC];

xi_init     = [1,0,Tamb,xSEI0]';
V_min       = 0;

%% Run simulation
out = sim(in, 'ShowProgress' ,true); % array
% Only Run for EqC
if mdl == 'Bat_EQC'
    
    Vt      = squeeze(out.EC_Vt.Data);
    T       = squeeze(out.EC_T.Data)-273.15;
    xSEI    = out.Results(:,5);
    P       = out.Results(:,7);
    t       = out.tout;
    I       = out.Results(:,9)./4.6;
end

% Only Run for SPMe
if mdl == 'Bat_SPM'
    Vt      = squeeze(out.V.Data);
    T       = squeeze(out.T.Data)-273.15;
    xSEI    = squeeze(out.x_sei.Data);
    P       = squeeze(out.P.Data)./1e3;
    t       = out.tout;
    I       = squeeze(out.I.Data)./4.6;
end
%% Plot
set(figure(1),'position',[100 200 500 700])

figure(1),
tiledlayout(5,1,'TileSpacing','tight', 'TileIndexing','columnmajor');

ax1 = nexttile(1);
plot(t, Vt), grid on
ylabel('Measured V_t [V]')
xlabel('Time [s]')
ylim([-0.1,4.1])

ax2 = nexttile(2);    
plot(t,T), grid on
hold on,
yline(50,'k--','LineWidth',2)
ylabel('Temp. [degC]')
xlabel('Time [s]')
legend("T_{bat}","T_{set}")

ax3 = nexttile(3);    
plot(t,xSEI), grid on
ylabel('x_{SEI} [-]')
xlabel('Time [s]')

ax4 = nexttile(4);    
plot(t,P), grid on
ylabel('P [kPa]')
xlabel('Time [s]')

ax5 = nexttile(5);    
plot(t,I), grid on
ylabel('Discharge C-Rate')
xlabel('Time [s]')

linkaxes([ax1,ax2,ax3,ax4,ax5],"x")
xlim([0,t(end)])
nexttile(2)
set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',10)
