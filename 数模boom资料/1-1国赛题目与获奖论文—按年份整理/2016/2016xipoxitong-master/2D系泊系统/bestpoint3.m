function [besty0, bestx0] = bestpoint3(H, x0, v_wind, m_qiu, I, L, xitong_figure)
%�˺�����fzero�������ų�ˮ���h
% 
%%%%����%%%%
% H����ˮ���
% x0�������ʼ������
% v_wind������
% m_qiu������������
% I ��ê���ͺ�
% L��ê������
% y0_yn_figure���Ƿ����y0��yn�ĺ���ͼ��
%
%%%%���%%%%
% besty0����������������
% bestx0���������ź�����
%
%%%%�˳������������������%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ĳ�ʹ���ڵ�ѡ��II�͵纸ê��22.05m��
% ѡ�õ������������Ϊ1200kg��
% �ֽ����ʹ���ڵ㲼����ˮ��18m��
% ����ƽ̹����ˮ�ܶ�Ϊ1.025��103kg/m3�ĺ���
% ����ˮ��ֹ���ֱ���㺣�����Ϊ12m/s��24m/sʱ��Ͱ��
% ���ڸֹܵ���б�Ƕȡ�ê����״������ĳ�ˮ��Ⱥ��ζ�����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

fun = @(y0)bestpoint3fun(y0, H, x0, v_wind, m_qiu, I, L, xitong_figure);
y0 = -0.3; % initial point
besty0 = fzero(fun, y0);
[~, x, ~, ~, ~] = For2D(besty0, x0, v_wind, m_qiu, I, L, xitong_figure);
bestx0 = x0 - x(end);
end

function f = bestpoint3fun(y0, H, x0, v_wind, m_qiu, I, L, xitong_figure)
%�˺������ڹ���fzero���������룬�����yn = -H��
%

[y, ~, ~, ~, ~] = For2D(y0, x0, v_wind, m_qiu, I, L, xitong_figure);
yn = y(end);
f = yn - (-H);
end










