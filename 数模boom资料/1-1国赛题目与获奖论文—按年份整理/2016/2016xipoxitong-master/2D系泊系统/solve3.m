%% ���ļ������������ʣ�����m_qiu��L��Iʹ��һĿ����С
%% �Ż�����
%��������
clc, clear
I = 5;
c1 = 1;
c2 = 1;
v_wind = 24;
H = 18;
N = 500;
x0 = 20;
y0_yn_figure = 0;
xitong_figure = 0;

%Ŀ�꼰Լ��
fun = @(x) GA_m_l(x, I, c1, c2, v_wind, H, N, x0, y0_yn_figure, xitong_figure);
A = [];
b = [];
Aeq = [];
beq = [];
lb = [0, H-5];
ub = [inf, inf];
nonlcon = @(x)circlecon_m_l(x, I, v_wind, H, N, x0, y0_yn_figure, xitong_figure);

%% ����GA�㷨��˷������Ż�-----ʧ����
% nvars = 2;         % ����ı�����Ŀ
% options = gaoptimset('PopulationSize',100,'CrossoverFraction',0.75,'Generations',20,'StallGenLimit',40,'PlotFcns',{@gaplotbestf,@gaplotbestindiv}); %��������
% [x_best, fval,  exitflag] = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, options);   

%% ����fmincon��˷������Ż������з�����Լ���ģ�
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
X0 = [0, 20];
x_m_l = fmincon(fun, X0, A, b, Aeq, beq, lb, ub, nonlcon, options);

%���ƽ��
m_qiu = x_m_l(1);
L = x_m_l(2);
x0 = 20;
xitong_figure = 1;
[besty0, bestx0] = bestpoint(H, N, x0, v_wind, m_qiu, I, L, y0_yn_figure);
y0 = besty0;
x0 = bestx0;
[y1, x1, theta1, T1, stat1] = For2D(y0, x0, v_wind, m_qiu, I, L, xitong_figure);


