clear; clc; close all;

%% Load and preprocess data

% calculate heat gen
data0 = readtable('UMBL2022FEB_CELL152083_THERMPULSE_P25C_20240716_R0_normal.csv'); % initial *.res filename mistaken for CELL152049
i_start = 1; % start of data to fit (e.g. |I|>5A) find(abs(data0.current)>5, 1, 'first')-1000;
data = data0(i_start:end,:);m
t_data = data.test_time-data.test_time(1);
I = data.current; 
V =data.voltage;
V_ocv = mean([3.64246, 3.63785]); %V from voltage plot when at rest in middle and end of test
Q_dot = I.*(V-V_ocv);

% plot heat gen
% figure(1)
% plot(t,I)
% xlabel('Time [s]')
% ylabel('Q=I(V-V$_\mathrm{ocv}$) [W]')
% set(findall(gcf,'type','line'),'linewidth',2)
% set(findall(gcf,'type','axes'),'fontsize',11)

% denoise/filter T data 
T_center0 = data.aux_0_u_C;
T_tab0 = data.aux_1_u_C;
T_fixture0 = data.aux_2_u_C;
T_amb0 = data.aux_3_u_C; 
order = 1;
framelen = 301;
T_center = sgolayfilt(T_center0, order, framelen);
T_tab = sgolayfilt(T_tab0, order, framelen);
T_fixture = sgolayfilt(T_fixture0, order, framelen);
T_amb = sgolayfilt(T_amb0, order, framelen);

% plot T filtering results
figure(2)
hold on 
plot(t_data,T_center0, t_data,T_tab0,t_data,T_fixture0, t_data,T_amb0)
plot(t_data,T_center, t_data,T_tab,t_data,T_fixture, t_data,T_amb) %LineStyle="--"
xlabel('Time [s]')
ylabel('Temperature [$^\circ$ C]')
xlim([0,t_data(end)])
legend(["$T_\mathrm{center, data}$", "$T_\mathrm{top, data}$", "$T_\mathrm{fixture, data}$","$T_\mathrm{amb, data}$",...
    "$T_\mathrm{center}$", "$T_\mathrm{top}$", "$T_\mathrm{fixture}$","$T_\mathrm{amb}$"],...
    Location="northwest")
set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',11)

%% Fit thermal paramters
% default initial guess (actually guessing for all masses)
C_th_cell_0 = 1100*0.2; % pybamm mCp = [J.kg-1.K-1]*[kg]
R_poron_0 = 0.00158/(0.09*(0.009*2)); %L/(k.A) where k=0.292 from PORON® 4701-V0-M Medium, k=0.09 from PORON® 4701-50 Firm
C_th_fixture_0 = 897*(10/2.2); % mCp [J/(kg.K)]*[kg]
R_conv = 1/(15*0.05); % [1/(h.A)] arbitrary
 
% set up ICs
p0 = [C_th_cell_0 R_poron_0 C_th_fixture_0 R_conv]';
T_data = [T_center, T_fixture];
T0 = T_data(1,:)';

% sanity check initial guess by simulating thermal model
tspan=[0,t_data(end)];
[t_sim, T_hat] = ode45(@(t,T)thermal_cell_fixture_model(t,p0,T,t_data, Q_dot,T_amb),tspan,T0);

% plot model results with initial guess
figure(3)
ylabels=["$T_\mathrm{center}$ [$^\circ$ C]", "$T_\mathrm{fixture}$ [$^\circ$ C]"];
for i = 1:size(T_data,2)
    subplot(size(T_data,2),1,i)
    hold on 
    plot(t_data, T_data(:,i),DisplayName= 'Data', Color='k')
    plot(t_sim, T_hat(:,i), DisplayName='Initial guess')
    ylabel(ylabels(i))
    xlim([0,t_data(end)])
    legend(Location='northwest')
end
xlabel('Time [s]')
set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',11)

%% fit parameters to data
% Set bounds to allow magnitude change from p0
ub = 10.*p0;
lb = 0.1.*p0;

% set up and solve optimization problem (8 mins)
rng default % for reproducibility
obj = @(p)objective(p, t_data, T_data, Q_dot, T_amb); 
tic;
options = optimoptions('patternsearch','Display','iter','PlotFcn',@psplotbestf); % show optimization progress
p = patternsearch(obj,p0,[],[],[],[],lb,ub, [],options); 
toc % display elapsed time 

%% plot fit results
[t_sim, T_hat] = ode45(@(t,T)thermal_cell_fixture_model(t,p,T,t_data, Q_dot,T_amb),tspan,T0);

figure(3)
hold on 
for i = 1:size(T_data,2)
    subplot(size(T_data,2),1,i)
    hold on 
    plot(t_sim, T_hat(:,i), DisplayName='Fit')
    ylabel(ylabels(i))
    xlim([0,t_data(end)])
    legend()
end
set(findall(gcf,'type','line'),'linewidth',2)

% display values and relative change in params compared to initial guess
p./p0 % check closeness of initial guess

C_th_cell = p(1)
R_poron = p(2)
C_th_fixture = p(3)
R_conv = p(4)

