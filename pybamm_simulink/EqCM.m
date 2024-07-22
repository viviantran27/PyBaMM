function xdot = EqCM(x,I)
    z = x(1);
    V1 = x(2);
    T = x(3); 
    x_sei = x(4); 

    % ELECTRICAL 
    Q = 4.6; %A.h
    alpha = 64.526;
    beta = 0.48053;
    z=0.5; % remove SOC dependence
    Rs = [0.051109270053841  -0.124390708105661...
            0.112323360710586  -0.046901650715843 0.015533047683140]...
            *z.^(4:-1:0)';
    R1 = [0.030297671277692  -0.078249784854926...
        0.076106125220171  -0.035096138367922   0.013959006299232]...
        *z.^(4:-1:0)'*alpha;
    C1 = [-0.178332712122388   0.724802617995660 ...
        -1.060271845888578   0.759649001977427   0.210118417436246]...
        *1e4*z.^(4:-1:0)'*beta;
    
    S = diag([0 -1/(R1*C1)]);
    xdot_e = S*[z V1]' + [-1/Q/3600 1/C1]'*I; 



    % SEI DECOMPOSITION
    A_sei = 2.25e15; %s-1
    E_sei = 2.24e-19; %J
    k_b = 1.3806e-23; %J.K-1
%     x_sei_0 = 0.15; %initial fraction of Li in SEI
    x_sei_dot = -A_sei*exp(-E_sei/k_b/T)*x_sei;


    % THERMAL
    hA = 0.4865;
    mCp =  219.2578;
    T_amb = 273.15+25; % K
    % m_an = 0.0191;
    h_sei =257000;
    Q_sei = m_an*h_sei*x_sei_dot;
    Tdot = 1/mCp*(I*(V1+Rs*I) + Q_sei)- hA/mCp*(T-T_amb); %redefine deltaV?
   
    % Recombine states
	xdot = [xdot_e;Tdot;x_sei_dot];
end