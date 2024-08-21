function [bestz0, besty0, bestx0] = bestpoint1_3D(H, y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure)
%�˺�����fzero�������ų�ˮ���h
% 
%%%%����%%%%
% H����ˮ���
% y0�������ʼ������
% x0�������ʼ������
% v1������
% v2������
% m_qiu������������
% I ��ê���ͺ�
% L��ê������
% xitong_figure���Ƿ����ϵͳͼ��
%
%%%%���%%%%
% bestz0����������z����
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

fun = @(z0)bestpoint3_3Dfun(z0, H, y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure);
z0 = -0.3; % initial point
bestz0 = fzero(fun, z0);
[~, y, x, ~, ~, ~, ~] = For3D(bestz0, y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure);
besty0 = y0 - y(end);
bestx0 = x0 - x(end);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = bestpoint3_3Dfun(z0, H, y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure)
%�˺������ڹ���fzero���������룬�����zn = -H��
%

[z, ~, ~, ~, ~, ~, ~] = For3D(z0, y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure);
zn = z(end);
f = zn - (-H);
end