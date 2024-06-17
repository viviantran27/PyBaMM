clear all; close all; 
%warning off; 
clc; tic;
p = py.sys.path;
p.insert(int32(0),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM') %path to pybamm package
p.insert(int32(1),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\venv\Lib\site-packages') % path to other libraries in venv
pybamm = py.importlib.import_module('pybamm');

% Regenerate the casadi objects
py.pybamm_setup.main()
addpath('C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink\casadi-windows-matlabR2016a-v3.5.5')  
delete(gcp('nocreate'))
disp([datestr(now, 'HH:MM:SS'),'  Model set up']);

% The simulink model name
mdl = 'pack_2s1p';

% Read in settings and calculate some derived parameters
param = table2struct(readtable('settings.csv'));
param.I_mag = 2;
param.battery_volume = param.battery_height*pi*(param.battery_radius)^2;
param.battery_mass = param.battery_volume*param.rho;
param.cooled_surface_area = param.battery_height*2*pi*param.battery_radius;
param.t_init = param.reference_temperature;
param.I_app = param.I_mag * param.I_sign_init;

% set drive cycle
param.drive_cycle = 'sflip';

% initial guesses for ocv and r_int
param.ocv_init = 4.0;
param.r_int_init = 0.1;

% Run the simulation
N_sum = 1;
in(1:N_sum) = Simulink.SimulationInput(mdl);
disp([datestr(now, 'HH:MM:SS'),'  Start simulink']);
parsimOut = sim(in, 'ShowProgress' ,true);
disp([datestr(now, 'HH:MM:SS'),'  Finish simulink']);
toc;

parsimOut.ErrorMessage
