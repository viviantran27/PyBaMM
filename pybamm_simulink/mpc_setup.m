
Ts = 0.1;

ss(A,B,C,D,Ts)
p = 20;
m = 10;
mpcobj = mpc(tf(1,[1 0 0]),Ts,p,m);