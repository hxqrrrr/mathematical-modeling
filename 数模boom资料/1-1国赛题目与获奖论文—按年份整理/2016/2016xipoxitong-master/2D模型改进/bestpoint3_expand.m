function bestxx = bestpoint3_expand(H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save)
%�˺����������ϵ��ϵͳ��3Ԫ�����飬�Ӷ��������y0,x0,alpha2��
%
%%%%����˵��%%%%
% H = 18;
% v1 = 24;%���� m/s
% v2 = 1.5;%ˮ�� m/s
% m_qiu = 1200;%���������� kg
% I = 2;
% L = 22.05;
% xitong_figure = 0;%�����ŵ�ʱ = 0������ϵͳʱ = 1��
% xitong_save = 0;
%

%%%%����%%%%
fun = @(xx)For2D_expand(xx, H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);
xx0 = [H-0.7, 20, 0];
bestxx = fsolve(fun, xx0);
end




















