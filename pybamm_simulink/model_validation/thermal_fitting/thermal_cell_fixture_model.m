function [T_dot] = thermal_cell_fixture_model(t,p,T,t_data, Q_dot_data,T_amb_data)
%THERMAL_CELL_FIXTURE_MODEL 2-state thermal ODE assuming all cell heat
%loss is through the fixture. Uses time-varying input from data. 

% interpolate input for time t
Q_dot = interp1(t_data, Q_dot_data, t);
T_amb = interp1(t_data, T_amb_data, t);

% separate parameters
C_th_cell = p(1);
R_poron = p(2);
C_th_fixture = p(3);
R_conv = p(4);

% calculate dT/dt
A = [1/C_th_cell 1/C_th_fixture]'.*[-1/R_poron 1/R_poron;... 
    1/R_poron -(1/R_poron + 1/R_conv)];
B = [1/C_th_cell 0; ...
    0 1/(R_conv*C_th_fixture)];
u = [Q_dot T_amb]';
T_dot = A*T + B*u; % [T_cell T_fixture]'
end

