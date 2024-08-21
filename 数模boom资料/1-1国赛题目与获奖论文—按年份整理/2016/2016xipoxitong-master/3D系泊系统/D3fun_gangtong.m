function y = D3fun_gangtong(x_point, T5, theta5, alpha5, beta, v2, m_qiu)
%�˺�����������Ͱ��T��theta��alpha
%
%%%%%����˵��%%%%
% x���⡣T,theta��alpha
% T��Ti-1
% theta��



%%%%����%%%%
Tx = T5*cos(theta5)*cos(alpha5);
Ty = T5*cos(theta5)*sin(alpha5);
Tz = T5*sin(theta5);

%��Ͱ��������
rho = 1.025*10^3;
g = 9.8;
m_tong = 100;%��Ͱ������ kg
G_tong = m_tong*g;%��Ͱ����
% m_qiu = 1200;%���������� kg
G_qiu = m_qiu*g;%����������
l_tong = 1;%��Ͱ�� m
D_tong = 30/100;%��Ͱ�׳�
F_tong = rho*g*pi*(D_tong/2)^2*l_tong;%��Ͱ����

% s = D_tong*(l_tong^2 - l_tong^2*(cos(theta5))^2*(sin(alpha5-beta))^2);
s = D_tong*l_tong*sqrt((cos(theta5))^2*(cos(beta - alpha5))^2 + (sin(theta5))^2);
Fs = 374*s*v2^2;%ˮ����

Ti = x_point(1);
thetai = x_point(2);
alphai = x_point(3);

Tix = Ti*cos(thetai)*cos(alphai);
Tiy = Ti*cos(thetai)*sin(alphai);
Tiz = Ti*sin(thetai);

y = [Tix - Tx - Fs*sin(beta);...
    Tiy - Ty - Fs*cos(beta);...
    Tiz + G_tong + G_qiu  - F_tong - Tz];
end