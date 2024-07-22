clear; close all; 
%warning off; 
clc; 
p = py.sys.path;
p.insert(int32(0),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM') %path to pybamm package
p.insert(int32(1),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\venv\Lib\site-packages') % path to other libraries in venv

% p.insert(int32(2),'C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink\model_validation')
addpath('C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink')  
addpath('C:\Users\Vivian\University of Michigan Dropbox\Vivian Tran\from_box\Research\PyBaMM\PyBaMM\pybamm_simulink\casadi-windows-matlabR2016a-v3.5.5')  

pybamm = py.importlib.import_module('pybamm');

% Regenerate the casadi objects
cell_names = ["100A", "100B", "75", "50"]';
param = table2struct(readtable('sim_settings.csv'));
param.soc_init = 0.50;
param.soc_name = '50';
writetable(struct2table(param),'sim_settings.csv');
py.pybamm_setup_validation.main() % rerun pybamm with updated soc
% delete(gcp('nocreate'))
disp([datestr(now, 'HH:MM:SS'),'  Model set up']);

% The simulink model name
mdl ='SPMe_ESC_validation_simulink'; 

% Read in settings and calculate some derived parameters
% param = table2struct(readtable('sim_settings.csv'));
% param.I = 5*25; % I>0 for discharge
p0 = [0.114319142120841  -0.663997904709863 ...
        1.686768366470837  -2.462409468590447   2.282769243949050 ...
        -1.401600551327618   0.576551904459066  -0.157044814050174 ...
        0.027351974716897  -0.002866912229499   0.000171009483610...
        0.000029999998915].*1e5;
param.ocv_init = polyval(p0, param.soc_init);   % initial cell voltage
param.endtime = 55;
V_min = 0;
R_tabs = dictionary(cell_names, [0.0086,0.0049,0.0078+0.0005, 0.007]');
R_escs = dictionary(cell_names, [0.0067,0.004,0.0067-0.0005, 0.0067]');
R_tab = R_tabs(param.soc_name);
R_esc = R_escs(param.soc_name);

% Load venting parameters and T_amb
% load('venting_params.mat')

% thermal model params (internal)
% A_cool = 0.025549; % Cell cooling surface area [m2] pouch 0.41 default Real dim ~[130mm×89mm×5.5mm] 
% V_cell = 6.3635E-5; % Cell volume [m3],pouch 3.92E-5 Real dim ~[130mm×89mm×5.5mm] 
% rho_Cp_eff = 6271195.03973258; 
% Cp = rho_Cp_eff/ (1100*2.4847642158614396); % hack. Cp from electrode, not cell
% m_cell = rho_Cp_eff/Cp*V_cell;
% h = 34.83874617168782;
% T_amb = 25+273.15;


% Run the simulation
tic;
N_sum = 1;
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
pybamm_sol = readtable("pybamm_ESC_sim_results\pybamm_" + param.soc_name +"_ESC_sim_results.csv");

figure(1)
t=sol.tout;
I = squeeze(sol.I.Data);
V = squeeze(sol.V.Data);
T = squeeze(sol.T.Data);
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
xlim([0,60])
legend

nexttile;
hold on 
plot(pybamm_sol.t, pybamm_sol.V, '-k')
plot(t, V, '--r')
ylabel('Measured voltage [V]')
xlabel('Time [s]')
xlim([0,60])
ylim([-2,2.5])

nexttile;
hold on 
plot(pybamm_sol.t, pybamm_sol.T-273.15, '-k')
plot(t, T-273.15, '--r')
xlim([0,60])
ylabel('Temperature [degC]')
xlabel('Time [s]')

% nexttile;
% plot(x, ce(:,1:30:length(ce)))
% ylabel('$c_\mathrm{e}$')
% legend(string(t(1:30:length(ce))) + 's', NumColumns=3, Location="ne")
% xlabel('x')
% 
% nexttile;
% plot(r, cs_n(:,1:100:length(cs_n)))
% ylabel('$c_\mathrm{s,n}$')
% xlabel('r')
% 
% nexttile;
% plot(r, cs_p(:,1:100:length(cs_p)))
% ylabel('$c_\mathrm{s,p}$')
% xlabel('r')

set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',12)
set(gcf,'Position',[40 60 900,900 ])
