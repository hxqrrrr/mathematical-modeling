function [c, ceq] = circlecon_m_l_expand(x, I, v1, v2, H, xitong_figure)
%�˺����ǵ��������m_qiu��L�Ż�����ķ�����Լ��
%

%%%%����%%%%
m_qiu = x(1);
L = x(2);

xitong_save = 0;
bestxx = bestpoint3_expand(H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%�����ŵ�
xitong_save = 1;
[~] = For2D_expand(bestxx, H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%����ϵͳ
load('ϵͳ��Ϣ.mat', 'stat')
alpha1 = stat.alpha1;
alpha2 = stat.alpha2;
L_tuo = stat.L_tuo;
h = stat.h;

%������Լ��
rho = 1.025*10^3;%��ˮ���ܶ�  kg/m^3
D = 2;%Բ���������ֱ�� m
m0= 1000;%�������� kg
h_min = (m0+m_qiu)/(rho*pi*(D/2)^2);

c(1) = alpha1 - 5;
c(2) = -alpha1;
c(3) = alpha2 - 16;
c(4) = -alpha2;
c(5) = h - 2;
c(6) = -(h - h_min);
c(7) = L_tuo - 0.3;

ceq = [];%ceq = L_tuo;
end























