clear; close all; 
%warning off; 
clc; 
p = py.sys.path;
p.insert(int32(0),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM') %path to pybamm package
p.insert(int32(1),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\venv\Lib\site-packages') % path to other libraries in venv
p.insert(int32(2),'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\venv\Lib\site-packages\casadi')
pybamm = py.importlib.import_module('pybamm');

% Regenerate the casadi objects
% py.pybamm_setup.main()
% delete(gcp('nocreate'))
addpath('.\casadi-windows-matlabR2016a-v3.5.5')  
disp([datestr(now, 'HH:MM:SS'),'  Model set up']);

% The simulink model name
mdl ='SPMe_fast_discharge';

% Read in settings and calculate some derived parameters
ocv_init = 4.2;   % initial cell voltage
t_end = 60*8;
dt = 0.2; % needs to be the same as when generated pybamm casadi objects 
V_min = 0;

% Run the simulations
C_rates = 5:5:40;
for c=1:length(C_rates)
    tic;
    Q_nom = 4.6;
    I = 4.6*C_rates(c);
    N_sum = 1;
    in(1:N_sum) = Simulink.SimulationInput(mdl); 
    disp([datestr(now, 'HH:MM:SS'),'  Start simulink']);
    sols(c) = sim(in, 'ShowProgress' ,true);
    disp([datestr(now, 'HH:MM:SS'),'  Finish simulink']);
    toc;

    % Display error message
    if ~isempty(sols(c).ErrorMessage)
        sols(c).ErrorMessage
    end
end

%% plot results and compare to pybamm 
figure(1)
tiledlayout(2,1,'TileSpacing','tight', 'TileIndexing','columnmajor');

% plot signals
for i =1:length(sols)
    sol=sols(i);

    t = sol.tout;
    V = squeeze(sol.V.Data);
    T = squeeze(sol.T.Data);
    
    ax1 = nexttile(1);
    hold on 
    plot(t, V, DisplayName= string(C_rates(i))+'C')
    ylabel('Measured voltage [V]')
    xlabel('Time [s]')
    ylim([-0.1,4])
    
    ax2 = nexttile(2);
    hold on 
    plot(t, T-273.15)
    ylabel('Temperature [degC]')
    xlabel('Time [s]')
    ylim([20,70])
end

% format plot
linkaxes([ax1,ax2],"x")
xlim([0,t_end])
nexttile(1)
legend(NumColumns=2, Location="se")
set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',10)
% set(gcf,'Position',[40 60 900,1000 ])
