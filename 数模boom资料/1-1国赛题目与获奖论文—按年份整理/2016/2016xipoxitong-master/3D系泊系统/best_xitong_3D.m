%% ����bestx0, besty0��bestz0����µ�ϵͳ��Ϣ��ϵͳ3Dͼ��
clc
clear
%����Ϊ12ʱ��ϵͳ���
H = 18;
N = 100;
y0 = 20;
x0 = -5;
v1 = 12;
v2 = 1.5;
m_qiu = 500;%����Ӱ��ϵͳ
I = 2;
L = 22.05;
beta = pi/2;
z0_zn_figure = 1;
xitong_figure = 1;

[bestz0, besty0, bestx0] = bestpoint_3D(H, N, y0, x0, v1, v2, m_qiu, I, L, beta, z0_zn_figure);
z0 = bestz0;
y0 = besty0;
x0 = bestx0;
[z1, y1, x1, theta1, alpha, T1, stat1] = For3D(z0, y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure);

%����Ϊ24ʱ��ϵͳ���
v_wind = 24;
[besty0, bestx0] = bestpoint(H, N, x0, v_wind, m_qiu, I, L, y0_yn_figure);
y0 = besty0;
x0 = bestx0;
[y2, x2, theta2, T2, stat2] = For2D(y0, x0, v_wind, m_qiu, I, L, xitong_figure);  

%����Ϊ36ʱ��ϵͳ���
v_wind = 36;
[besty0, bestx0] = bestpoint(H, N, x0, v_wind, m_qiu, I, L, y0_yn_figure);
y0 = besty0;
x0 = bestx0;
[y3, x3, theta3, T3, stat3] = For2D(y0, x0, v_wind, m_qiu, I, L, xitong_figure);

%% ����fzero����bestx0, besty0��bestz0����µ�ϵͳ��Ϣ��ϵͳͼ��
clc
clear
%����Ϊ12ʱ��ϵͳ���
H = 18;
y0 = 20;
x0 = -5;
v1 = 12;
v2 = 1.5;
m_qiu = 500;%����Ӱ��ϵͳ
I = 2;
L = 22.05;
beta = pi/2;
xitong_figure = 0;
[bestz0, besty0, bestx0] = bestpoint1_3D(H, y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure);
xitong_figure = 1;
[z, y, x, theta, alpha, T, stat] = For3D(bestz0, besty0, bestx0, v1, v2, m_qiu, I, L, beta, xitong_figure);
% ע��fzero����������fsolve��������



