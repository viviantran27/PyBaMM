clear; close all; clc; 

% casadi path setup
p = py.sys.path;
% p.insert(int32(0), 'c:\users\vivian\anaconda3\lib\site-packages (3.5.5)')% path to python casadi 
p.insert(int32(0),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\venv\Lib\site-packages') % path to python casadi
addpath('C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink\casadi-windows-matlabR2016a-v3.5.5') % path to matlab casadi 
casadi_solver = py.importlib.import_module('casadi'); % import python casadi

% Model setup and sim
disp(string(datetime, 'HH:mm:SS')+'  Model set up');
% The simulink model name
mdl ='SPMe_EqCM';

% Read in settings and calculate some derived parameters
t_end = 60*8;
param = table2struct(readtable('sim_settings.csv'));
dt = param.dt; % needs to be the same as when generated pybamm casadi objects 
V_min = 0;
% SOC_OCV = [0:0.01:1]';
p0 = [0.114319142120841  -0.663997904709863 ...
        1.686768366470837  -2.462409468590447   2.282769243949050 ...
        -1.401600551327618   0.576551904459066  -0.157044814050174 ...
        0.027351974716897  -0.002866912229499   0.000171009483610...
        0.000029999998915].*1e5;
% OCV = (p0*SOC_OCV.^[11:-1:0]')';
Rs = 0.0078;
R_tab = 0.0086;
% discretize EqCM
Q_nom = 4.6;
SOC_0 = 1;
x0 = [SOC_0, 0, 25+273.15, 0.15]; %z, V1, T, x_sei


% Run the simulations
C_rates = 5:10:40;
for c=1:length(C_rates)
    tic;
    I = 4.6*C_rates(c);
    N_sum = 1;
    in(1:N_sum) = Simulink.SimulationInput(mdl); 
    disp(string(datetime, 'HH:mm:SS')+'  Start simulink');
    sols(c) = sim(in, 'ShowProgress' ,true); % array
    disp(string(datetime, 'HH:mm:SS')+'  Finish simulink');
    t_eval = toc
    disp(string(round(sols(c).tout(end)/t_eval, 2)) + ' sim to real time')

    % Display error message
    if ~isempty(sols(c).ErrorMessage)
        sols(c).ErrorMessage
    end
end