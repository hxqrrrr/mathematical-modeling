function [besty0, bestx0, bestyn] = bestpoint2(y0, x0, H, eta, maxt, eps, v_wind, m_qiu, I, L)
%�˺����õ����㷨�����ų�ˮ���h
% 
%%%%����%%%%
% y0����ʼ����������
% x0����ʼ�����ʼ������
% H��ˮ��
% eta��ѧϰ��
% maxt������������
% eps����⾫��
% v_wind������
% m_qiu������������
% I ��ê���ͺ�
% L��ê������
%
%%%%���%%%%
% besty0����������������
% bestx0���������ź�����
% bestyn��ê��ĩ��������
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

%%%%Ĭ������%%%%
if nargin < 10
    L = 22.05;%ê������ m
end
if nargin < 9
    I = 2;
end
if nargin < 8
    m_qiu = 1200;
end
if nargin < 7
    v_wind = 12;
end
if nargin < 6
    eps = 0.01;
end
if nargin < 5
    maxt = 500;
end
if nargin < 4
    eta = 0.001;
end
if nargin < 3
    H = 18;
end
if nargin < 2
    x0 = 20;
end
if nargin < 1
    y0 = 2*rand;
end

%%%%����%%%%
xitong_figure = 0;
t = 0;
while t < maxt
    [~, ~, ~, ~, stat] = For2D(y0, x0, v_wind, m_qiu, I, L, xitong_figure);
    yn = stat.yn;
    xn = stat.xn;
    delta_yn = yn - (-H);
    if abs(delta_yn) < eps
        disp('yn���㾫�ȣ���ֹ')
        break;
    else
        y1 = y0;
        y0 = y0 - eta*delta_yn;%����y0
        %���y0���ڷ�Χ��
        if y0 < -2 | y0 > 0
            eta1 = 0.5*eta;
            y0 = y1 - eta1*delta_yn;
        end
    end
    t = t+1;
    if t == maxt
        disp('�ﵽ��������������ֹ')
    end
end

%%%%�������%%%%
disp(['����������', num2str(t)])
besty0 = y0;
bestyn = yn;
bestx0 = x0 - xn;

























