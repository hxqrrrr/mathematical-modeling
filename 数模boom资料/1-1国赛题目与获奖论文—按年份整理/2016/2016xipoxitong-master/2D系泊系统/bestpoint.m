function [besty0, bestx0] = bestpoint(H, N, x0, v_wind, m_qiu, I, L, y0_yn_figure)
%�˺�������ɢö�ٷ������ų�ˮ���h
%
%%%%����%%%%
% H����ˮ���
% N��y0��[0,2]�����ɢ������������besty0����⾫�ȡ�
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

%Ĭ������
if nargin < 8
    y0_yn_figure = 0;
end
if nargin < 7
    L = 22.05;
end
if nargin < 6
    I = 2;
end
if nargin < 5
    m_qiu = 1200;
end
if nargin < 4
    v_wind = 12;
end
if nargin < 3
    x0 = 20;
end
if nargin < 2
    N = 2000;
end
if nargin < 1
    H = 18;
end

%%%%����%%%%

y0 = linspace(0, -2, N);
%ע��y0����ȡֵ��Χ�ģ�y0�����Դ�0��ʼȡֵ������ᷢ��yn>0�������
%����y0
rho = 1.025*10^3;%��ˮ���ܶ�  kg/m^3
D = 2;%Բ���������ֱ�� m
m0= 1000;%�������� kg
y0_min = -(m0+m_qiu)/(rho*pi*(D/2)^2);
y0 = linspace(y0_min, -2, N);

yn = zeros(size(y0));
xn = zeros(size(y0));
xitong_figure = 0;
for i = 1:length(y0)
    [y, x, theta, ~] = For2D(y0(i), x0, v_wind, m_qiu, I, L, xitong_figure);
    yn(i) =  y(end);
    xn(i) =  x(end);
    thetan(i) = theta(end - 1);
end

[~, ind1] = min(abs(yn - (-H)));
besty0 = y0(ind1);
bestx0 = x0 - xn(ind1);

%����y0��yn�ĺ���ͼ
if y0_yn_figure == 1
    figure
    plot(abs(y0), yn, 'r*-')
    xlabel('��ˮ���')
    ylabel('ê��ĩ������')
    title('ê��ĩ��yn���ˮ���h�ı仯����ͼ')

    figure
    plot(abs(y0), thetan, 'g*-')
    xlabel('��ˮ���')
    ylabel('ê��ĩ��ˮƽ�н�')
    title('ê��ĩ��ˮƽ�н����ˮ���h�ı仯����ͼ')
end
end










