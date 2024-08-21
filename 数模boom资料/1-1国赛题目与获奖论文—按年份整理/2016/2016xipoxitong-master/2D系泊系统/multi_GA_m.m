function f = multi_GA_m(m_qiu)
%�˺�����IENSGAii��Ŀ�꺯����
%
%��Ϊm_qiu
%Ŀ��1����ˮ�����С
%Ŀ��2���ζ�����͸�Ͱ�н���С
%

%����
%����
v_wind = 36;
%����
c = 10;
H = 18;
N = 500;
x0 = 20;
I = 2;
L = 22.05;
y0_yn_figure = 0;
xitong_figure = 0;

[besty0, bestx0] = bestpoint(H, N, x0, v_wind, m_qiu, I, L, y0_yn_figure);
[~, ~, ~, ~, stat] = For2D(besty0, bestx0, v_wind, m_qiu, I, L, xitong_figure);
alpha1 = stat.alpha1;

f(1) = abs(besty0);
f(2) = pi*bestx0^2 + c*alpha1;
end









