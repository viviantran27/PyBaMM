soc = linspace(-0.02,1.01, 100);
Dsn= 1.12070451*soc + 0.09209274;
Dsn2= 1.12070451*soc + 0.09209274;
close all
figure(1)
hold on 
plot(sto, Dsn, 'k')
plot(sto, Dsn2, '--r')
set(findall(gcf,'type','line'),'linewidth',2)
set(findall(gcf,'type','axes'),'fontsize',10)