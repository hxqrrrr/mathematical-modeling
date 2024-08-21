function [y, x, theta, T, stat] = For2D(y0, x0, v_wind, m_qiu, I, L, xitong_figure)
% �˺������ڸ���x0��y0�����ϵ��ϵͳ��״̬����
%
%%%%����%%%%
% y0�����������꣬|y0|=h�����У�hΪ��ˮ��ȡ�
% x0����������ꡣ
% v_wind�����١�
% m_qiu��������������
% I��ê���ͺš�1��2��3��4��5
% L��ê�����ȡ�
% outputfigure���Ƿ����ϵͳͼ��logistic
%
%%%%���%%%%
% y��ϵ��ϵͳ�����ꡣ����
% x��ϵ��ϵͳ�����ꡣ
% theta��ϵ��ϵͳ�Ƕȡ�
% T��ϵ��ϵͳ������
% stat��Ҫ���ϵ��ϵͳ��������������ˮ���h��������x0���ζ�����S����Ͱ��ֱ�н�alpha1��ê���׶�ˮƽ�н�alpha2������v_wind������������m��ϵͳ״̬yxthetaT��stats

%Ĭ������
if nargin < 7
    xitong_figure = 0;
end
if nargin < 6
    L = 22.05;%ê������ m
end
if nargin < 5
    I = 2;
end
if nargin < 4
    m_qiu = 1200;
end
if nargin < 3
    v_wind = 12;
end
if nargin < 2
    x0 = 20;
end
if nargin < 1
    disp('���������')
    return;
end

%%%%����%%%%
%ȷ��ê��
switch I 
    case 1
       II = 78/1000;%ê��ÿ�ڳ��� m
       m_II = 3.2;%��λ���ȵ����� kg/m
    case 2
       II = 105/1000;%ê��ÿ�ڳ��� m
       m_II = 7;%��λ���ȵ����� kg/m
    case 3
       II = 120/1000;
       m_II = 12.5;%��λ���ȵ����� kg/m
    case 4
        II = 150/1000;
        m_II = 19.5;%��λ���ȵ����� kg/m
    case 5
        II = 180/1000;
        m_II = 28.12;%��λ���ȵ����� kg/m
end
n = round(L/II);
ind = n+5+1;

y(1) = y0;
x(1) = x0; 
h = abs(y(1));%�����ˮ���

%��������
rho = 1.025*10^3;%��ˮ���ܶ�  kg/m^3
g = 9.8;%�������ٶ� N/kg
D = 2;%Բ���������ֱ�� m
h0 = 2;%Բ������߶� m
m0= 1000;%�������� kg
F0 = rho*g*pi*(D/2)^2*h;%���긡��
G0 = m0*g;%��������
%v_wind = 12;%���� m/s
S_wind = D*(h0-h);%��������� 
F_wind = 0.625*S_wind*v_wind^2;%����
theta1 = atan((F0-G0)/F_wind);%�ֹ�1��ˮƽ�н�
T1 = sqrt((F0-G0)^2+(F_wind)^2);%�ֹ�1������
T(1) = T1; theta(1) = theta1;

%�ֹ���������
for i = 1:4
    m(i) = 10;%�ֹ����� kg
    G(i) = m(i)*g;%�ֹ�����
    l(i) = 1;%�ֹܳ��� m
    d(i) = 50/1000;%�ֹ�ֱ�� m
    F(i) = rho*g*pi*(d(i)/2)^2*l(i);%�ֹܸ���
    
    T(i+1) = (  (F(i)-G(i)+T(i)*sin(theta(i)))^2  +...
        (T(i)*cos(theta(i)))^2    )^(1/2);
    theta(i+1) = atan(  (  (F(i)-G(i)+T(i)*sin(theta(i)))/...
        (T(i)*cos(theta(i)) ))   );
    
    %�ֹ�i�����꣨xi,yi��
    y(i+1) = y(i) - l(i)*sin(theta(i));
    x(i+1) = x(i) - l(i)*cos(theta(i));
end

%��Ͱ��������
m_tong = 100;%��Ͱ������ kg
G_tong = m_tong*g;%��Ͱ����
% m_qiu = 1200;%���������� kg
G_qiu = m_qiu*g;%����������
l_tong = 1;%��Ͱ�� m
D_tong = 30/100;%��Ͱ�׳�
F_tong = rho*g*pi*(D_tong/2)^2*l_tong;%��Ͱ����

T_tong = ( (F_tong-G_tong-G_qiu+T(5)*sin(theta(5)))^2 + ...
                (T(5)*cos(theta(5)))^2)^(1/2);
theta_tong = atan( ((F_tong-G_tong-G_qiu+T(5)*sin(theta(5)))...
                      /(T(5)*cos(theta(5)))) );
T(6) = T_tong;
theta(6) = theta_tong;

y(6) = y(5) - l_tong*sin(theta(5));
x(6) = x(5) - l_tong*cos(theta(5)); 

%ê���߷���
G_mao = II*m_II*g;%��λ��������
L_tuo = 0;%ê����β����
for i = 6 : 6+n-1
    if  theta(i) - 0 > 0.001
        T(i+1) = T(i) - G_mao*sin(theta(i));
        theta(i+1) = theta(i) - (G_mao*cos(theta(i)))/(T(i)-G_mao*sin(theta(i))); 
        y(i+1) = y(i) - sin(theta(i))*II;
        x(i+1) = x(i) - cos(theta(i))*II; 
    else 
        T(i+1) = 0;
        theta(i+1) = 0;
        y(i+1) = y(i);
        x(i+1) = x(i) - II;
        L_tuo = L_tuo+II;
    end
end

%�������[y, x, theta, T, stat]
if nargout == 5
    y = y;
    x = x;
    theta = theta;
    T= T; 
    % stat��Ҫ���ϵ��ϵͳ��������������ˮ���h��������x0���ζ�����S����Ͱ��ֱ�н�alpha1��ê���׶�ˮƽ�н�alpha2������v_wind������������m��ϵͳ״̬yxthetaT
    stat.h = abs(y0);
    stat.y0 = y0;
    stat.x0 = x0;
    stat.yn = y(end);
    stat.xn = x(end);
    stat.S = pi*x0^2;
    stat.alpha1 = 90 - 90*theta(5)/pi*2;%�Ƕ���
    stat.alpha2 = 90*theta(end-1)/pi*2;%�Ƕ���
    stat.v_wind = v_wind;
    stat.m_qiu = m_qiu;
    stat.L_tuo = L_tuo;
    stat.x = x;
    stat.y = y;
    stat.theta  =theta;
    stat.T = T;
end
if xitong_figure == 1
    figure
    %ϵͳ����
    plot(x(6:end), y(6:end), '-', 'color', rand(1, 3))
    hold on
    plot(x(1:5), y(1:5), '-*y', 'LineWidth', 2)
    plot(x(5:6), y(5:6), '-r', 'LineWidth', 12)
    %��Ư
    box_biao = [x0 - D/2, y0; x0+D/2, y0; x0+D/2, y0+h0; x0-D/2, y0+h0];
    fill(box_biao(:, 1), box_biao(:, 2), 'b');
    %������
    plot([x(6), x(6)], [y(6), y(6)-1.5], '-')
    A = linspace(0, 2*pi, 100);
    qiu_r = 0.25;%������뾶
    qiu_x = x(6)+qiu_r*sin(A);
    qiu_y = (y(6)-1.5)+qiu_r*cos(A);
    fill(qiu_x, qiu_y, 'k');
    %ê
    box_mao = [x(end), y(end); x(end)-1, y(end); x(end)-1, y(end)+1;
        x(end), y(end)+1];
    fill(box_mao(:, 1), box_mao(:, 2), 'k');
    %ע��
    alpha1 = 90 - 90*theta(5)/pi*2;%�Ƕ���
    alpha2 = 90*theta(end-1)/pi*2;%�Ƕ���
    text(x0-2, y0-1, [num2str(x0),',',num2str(y0)])
    text(x(6)-3, y(6)-2, ['����������:', num2str(m_qiu)])
    text(x(6)-0.5, y(6), ['��Ͱ���:', num2str(alpha1)])
    text(x(end)+0.5, y(end)+1.5, ['ê���н�:', num2str(alpha2)])
    if L_tuo ~= 0
        text(x(end)+0.5, y(end)+0.5, ['��β����:', num2str(L_tuo)])
    end
    %ͼ������
%     box off
    axis equal
    xlabel('��������')
    ylabel('��ֱ����')
    title(['����',num2str(v_wind),'����ˮ���', num2str(abs(y0)), 'ʱ��ϵ��ϵͳ'])
    hold off
end
    
end








