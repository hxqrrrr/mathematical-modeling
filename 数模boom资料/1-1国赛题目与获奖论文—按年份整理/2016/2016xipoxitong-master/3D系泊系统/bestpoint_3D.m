function [bestz0, besty0, bestx0] = bestpoint_3D(H, N, y0, x0, v1, v2, m_qiu, I, L, beta, z0_zn_figure)
%�˺�������ɢö�ٷ��������ų�ˮ���h
% 
%%%%����%%%%
% H����ˮ���
% N��z0��[0,2]�����ɢ������������besty0����⾫�ȡ�
% y0�������ʼ������
% x0�������ʼ������
% v1������
% v2��ˮ��
% m_qiu������������
% I ��ê���ͺ�
% L��ê������
% beta��������ˮ���ļнǡ�
% z0_zn_figure���Ƿ����y0��yn�ĺ���ͼ��
%
%%%%���%%%%
% bestz0����������z����
% besty0����������y����
% bestx0����������x����
%
%%%%�˳������������������%%%%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ���ڳ�ϫ�����ص�Ӱ�죬
% ���ź����ʵ��ˮ�����16m~20m֮�䡣
% ���ŵ�ĺ�ˮ�ٶ����ɴﵽ1.5m/s���������ɴﵽ36m/s��
% ��������Ƿ�����ˮ������ˮ������µ�ϵ��ϵͳ��ƣ�
% ������ͬ����¸�Ͱ���ֹܵ���б�Ƕȡ�ê����״������ĳ�ˮ��Ⱥ��ζ�����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%Ĭ������
if nargin < 11
    z0_zn_figure = 0;
end
if nargin < 10
    beta = pi/2;
end
if nargin < 9
    L = 22.05;
end
if nargin < 8
    I = 2;
end
if nargin < 7
    m_qiu = 500;
end
if nargin < 6
    v2 = 1.5;
end
if nargin < 5
    v1 = 12;
end
if nargin < 4
    x0 = -5;
end
if nargin < 3
    y0 = 20;
end
if nargin < 2
    N = 100;
end
if nargin < 1
    H = 18;
end

%%%%����%%%%

z0 = linspace(0, -2, N);
%ע��z0����ȡֵ��Χ�ģ�z0�����Դ�0��ʼȡֵ������ᷢ��zn>0�������
%����z0
rho = 1.025*10^3;%��ˮ���ܶ�  kg/m^3
D = 2;%Բ���������ֱ�� m
m0 = 1000;%�������� kg
z0_min = -(m0 + m_qiu)/(rho*pi*(D/2)^2);
z0 = linspace(z0_min, -2, N);

zn = zeros(size(z0));
yn = zeros(size(z0));
xn = zeros(size(z0));
thetan = zeros(size(z0));
alphan = zeros(size(z0));
xitong_figure = 0;
for i = 1:length(z0)
    [z, y, x, theta, alpha, ~, ~] = For3D(z0(i),y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure);
    zn(i) =  z(end);
    yn(i) =  y(end);
    xn(i) =  x(end);
    thetan(i) = theta(end - 1);
    alphan(i) = alpha(end - 1);
end

[~, ind1] = min(abs(zn - (-H)));
bestz0 = z0(ind1);
besty0 = y0 - yn(ind1);
bestx0 = x0 - xn(ind1);

%����y0��yn�ĺ���ͼ
if z0_zn_figure == 1
    figure
    plot(abs(z0), zn, 'r*-')
    xlabel('��ˮ���')
    ylabel('ê��ĩ������')
    title('ê��ĩ��zn���ˮ���h�ı仯����ͼ')
    
    figure
    plot(abs(z0), thetan, 'g*-')
    xlabel('��ˮ���')
    ylabel('ê��ĩ��ˮƽ��н�')
    title('ê��ĩ��ˮƽ��н����ˮ���h�ı仯����ͼ')    
end
end










