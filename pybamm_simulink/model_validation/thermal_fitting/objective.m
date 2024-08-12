function J = objective(p, t_data, T_data, Q_dot, T_amb)
    % solve model
    T0 = T_data(1,:)';
    tspan=[0,t_data(end)];
    [t, T_hat] = ode45(@(t,T)thermal_cell_fixture_model(t,p,T,t_data, Q_dot,T_amb),tspan,T0);
    J = mean((T_hat-interp1(t_data,T_data,t)).^2,"all"); % MSE
end