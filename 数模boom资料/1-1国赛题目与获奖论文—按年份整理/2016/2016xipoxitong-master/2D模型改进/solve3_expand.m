%% ���ļ������������ʣ�����m_qiu��L��Iʹ��һĿ����С
%% �Ż�����
%��������
clc, clear
I = 2;
c1 = 1;
c2 = 1;
v1 = 24;
v2 = 1.5;
H = 18;
xitong_figure = 0;

%Ŀ�꼰Լ��
fun = @(x)GA_m_l_expand(x, I, c1, c2, v1, v2, H, xitong_figure);
A = [];
b = [];
Aeq = [];
beq = [];
lb = [0, H-5];
ub = [inf, inf];
nonlcon = @(x)circlecon_m_l_expand(x, I, v1, v2, H, xitong_figure);

%% ����GA�㷨��˷������Ż�
% nvars = 2;         % ����ı�����Ŀ
% options = gaoptimset('PopulationSize',100,'CrossoverFraction',0.75,'Generations',20,'StallGenLimit',40,'PlotFcns',{@gaplotbestf,@gaplotbestindiv}); %��������
% [x_best, fval,  exitflag] = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, options);   

%% ����fmincon��˷������Ż������з�����Լ���ģ�
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
X0 = [1200, 28];
x_m_l = fmincon(fun, X0, A, b, Aeq, beq, lb, ub, nonlcon, options);

%���ƽ��
m_qiu = x_m_l(1);
L = x_m_l(2);

xitong_figure = 0;
xitong_save = 0;
bestxx = bestpoint3_expand(H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%�����ŵ�

xitong_figure = 1;
xitong_save = 1;
[~] = For2D_expand(bestxx, H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%����ϵͳ


