% Plot model results from EqCM and SPMe and compare
clear; clc;
close all   

% set matlab filepaths
addpath('EqCM_ESC_sim_results\')  
addpath('pybamm_ESC_sim_results\')
addpath('ESC_data\')


SOCs = [100, 75, 50];
t_max_inset = 60; 
Q_nom=4.6;

% LOAD AND PLOT DATA AND SIM RESULTS
figure(1)
tiledlayout(3,length(SOCs),'TileSpacing','none', 'TileIndexing','columnmajor');

for i = 1:length(SOCs)
    SOC = SOCs(i);
    
    % load pybamm sim results
    spme = readtable('pybamm_' + string(SOC) +'_ESC_sim_results.csv');
    
    %  load eqcm sim results
    eqcm_sim = load('ESC_fit_sim_results_' + string(SOC) +'.mat');
    eqcm = eqcm_sim.out;
    
    % load data
    % data = eqcm_sim.data; 
    data = readtable('ESC_' + string(SOC) +'SOC_full.csv');
    data = data(find(-data.CurrentShunt>10,1):end, :);
    data.t = data.Time_s_ - data.Time_s_(1);

    % PLOT SIM RESULTS
    ax(1+(i-1)*length(SOCs)) = nexttile(1+(i-1)*length(SOCs));
    hold on 
    plot(data.t, -data.CurrentShunt/Q_nom, 'k', DisplayName='Data')
    plot(eqcm.I.Time, eqcm.I.Data/Q_nom, '-.b', DisplayName='EqCM')
    plot(spme.t, spme.I/Q_nom, ':r', DisplayName='SPMe')
    % xlabel('Time [s]')
    set(gca,'Xticklabel',[])
    xlim([0,600])
    ylim([0, 55])
    title(string(SOC) + '\% SOC$_0$')
    if i==1
        ylabel('Current [A/A.h]')
    else
        set(gca,'Yticklabel',[])
    end


    ax(2+(i-1)*length(SOCs)) = nexttile(2+(i-1)*length(SOCs));
    hold on 
    plot(data.t, data.Voltage_V_, 'k', DisplayName='Data')
    plot(eqcm.V, '-.b', DisplayName='EqCM')
    plot(spme.t, spme.V, ':r', DisplayName='SPMe')
    % xlabel('Time [s]')
    set(gca,'Xticklabel',[])
    xlim([0,600])
    ylim([0, 2.3])
    if i==1
        ylabel('Voltage [V]')
    else
        set(gca,'Yticklabel',[])
    end


    ax(3+(i-1)*length(SOCs)) = nexttile(3+(i-1)*length(SOCs));
    hold on 
    plot(data.t, data.CellTemperature, 'k', DisplayName='Data')
    % plot(data.Time_s_, data.Thermocouple_2, 'k', DisplayName='Data')
    plot(eqcm.T_cell-273.15, '-.b', DisplayName='EqCM')
    plot(spme.t, spme.T-273.15, ':r', DisplayName='SPMe')
    xlabel('Time [s]')
    xlim([0,600])
    ylim([20, 140])
    if i==1
        ylabel('Temperature [$^\circ$C]')
    else
        set(gca,'Yticklabel',[])
    end
end

% create figure legend. will scoot figure
lg = legend(ax(6), NumColumns=3);
lg.Location = 'southoutside';

for j=1:length(SOCs)

    SOC = SOCs(j);
    
    % load pybamm sim results
    spme = readtable('pybamm_' + string(SOC) +'_ESC_sim_results.csv');
    
    %  load eqcm sim results
    eqcm_sim = load('ESC_fit_sim_results_' + string(SOC) +'.mat');
    eqcm = eqcm_sim.out;
    
    % load data
    % data = eqcm_sim.data; 
    data = readtable('ESC_' + string(SOC) +'SOC_full.csv');
    data = data(find(-data.CurrentShunt>10,1):end, :);
    data.t = data.Time_s_ - data.Time_s_(1);

    subax =  ax(1+3*(j-1));
    p = get(subax, 'Position');
    ax2= axes('Box','on','position',[ p(1)+p(3)*0.3, p(2)+p(4)*0.3, p(3)*.65, p(4)*.65]); % [x0 y0 width height]
    hold on 
    plot(data.t, -data.CurrentShunt/Q_nom, 'k', DisplayName='Data')
    plot(eqcm.I.Time, eqcm.I.Data/Q_nom, '-.b', DisplayName='EqCM')
    plot(spme.t, spme.I/Q_nom, ':r', DisplayName='SPMe')
    xlim([0,t_max_inset])
    ylim([0, 55])

    subax = ax(2+3*(j-1));
    p = get(subax, 'Position');
    ax2= axes('Box','on','position',[ p(1)+p(3)*0.3, p(2)+p(4)*0.3, p(3)*.65, p(4)*.65]); % [x0 y0 width height]
    hold on 
    plot(ax2, data.t(data.t<t_max_inset), data.Voltage_V_(data.t<t_max_inset), 'k', DisplayName='Data')
    plot(ax2, eqcm.V.Time(eqcm.V.Time<t_max_inset), eqcm.V.Data(eqcm.V.Time<t_max_inset), '-.b', DisplayName='EqCM')
    plot(ax2, spme.t(spme.t<t_max_inset), spme.V(spme.t<t_max_inset), ':r', DisplayName='SPMe')
    axis tight
    xlim([0,t_max_inset])
    ylim([0, 2.3])
end

set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',11)
set(gcf,'Position',[40 60 900,500])
