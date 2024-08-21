function f = For2D_expand(xx, H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save)
%�˺�����ϵ��ϵͳ��3Ԫ�������Ŀ�꺯��,����fsolve������ģ�͸Ľ�������"����ƽ��"��"�����߷���"
%
%%%%����%%%%
% xx��y0,x0,alpha2��
% v1�����١�
% v2��ˮ�١�
% H����ˮ��ȡ�
% m_qiu��������������
% I��ê���ͺš�1��2��3��4��5
% L��ê�����ȡ�
% xitong_figure���Ƿ�����ϵͳͼ��
% xitong_save���Ƿ񱣴�ϵͳ��ֵ���


%%%%����%%%%
y0 = xx(1);
x0 = xx(2);
alpha2 = xx(3);

h = H - y0;
y(1) = y0;
x(1) = x0;

%��������
rho = 1.025*10^3;%��ˮ���ܶ�  kg/m^3
g = 9.8;%�������ٶ� N/kg
D = 2;%Բ���������ֱ�� m
h0 = 2;%Բ������߶� m
m0= 1000;%�������� kg
F0 = rho*g*pi*(D/2)^2*h;%���긡��
G0 = m0*g;%��������
% v1 = 24;%���� m/s
% v2 = 1.5;%ˮ�� m/s

S_wind = D*(h0 - h);%��������� 
Fw = 0.625*S_wind*v1^2;%����
Fs0 = 374*D*h*v2^2;%��ˮ����

Tx(1) = Fw + Fs0;
Ty(1) = F0 - G0;

%�ֹ�����
for i = 1:4
    l(i) = 1;%�ֹܳ��� m
    d(i) = 50/1000;%�ֹ�ֱ�� m
    m(i) = 10;%�ֹ����� kg
    G(i) = m(i)*g;%�ֹ�����
    F(i) = rho*g*pi*(d(i)/2)^2*l(i);%�ֹܸ���

    %%%%�ο���http://blog.sina.com.cn/s/blog_53f291190100cjss.html
    si = @(theta1)d(i)*(pi/2*d(i)*cos(theta1) + l(i)*sin(theta1));%��һ���ֹܵĺ�ˮ��ƽ��ͶӰ
    Fsi = @(theta1)374*si(theta1)*v2^2;%��һ���ֹܵĺ�ˮ��

    fun = @(theta1)(G(i) - F(i))/2*cos(theta1) + Tx(i)*sin(theta1) + Fsi(theta1)*sin(theta1)/2 - Ty(i)*cos(theta1);
    theta1 = fsolve(fun, 0);
    theta(i) = theta1;
    
    Tx(i+1) = Tx(i) + Fsi(theta(i));
    Ty(i+1) = Ty(i) + F(i) - G(i);
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

si = @(theta1)D_tong*(pi/2*D_tong*cos(theta1) + l_tong*sin(theta1));%��һ���ֹܵĺ�ˮ��ƽ��ͶӰ
Fsi = @(theta1)374*si(theta1)*v2^2;%��һ���ֹܵĺ�ˮ��

fun = @(theta1)(G_tong - F_tong)/2*cos(theta1) + Tx(5)*sin(theta1) + Fsi(theta1)*sin(theta1)/2 - Ty(5)*cos(theta1);
theta_tong = fsolve(fun, 0);
theta(5) = theta_tong;

Tx(6) = Tx(5) + Fsi(theta(5));
Ty(6) = Ty(5) + F_tong - G_tong - G_qiu;
y(6) = y(5) - l_tong*sin(theta(5));
x(6) = x(5) - l_tong*cos(theta(5)); 

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
n = floor(L/II);
% L = 22.05;
w = m_II*g;
% if alpha2 == 0 & Ty(6) <= n*w
%     
%     f(1) = Tx(6)/w*cosh(w/Tx(6)*x(6) + log(tan(alpha2) + sec(alpha2))) - Tx(6)/w*sec(alpha2)...
%         - y0 + l(1)*sin(theta(1)) + l(2)*sin(theta(2)) + l(3)*sin(theta(3)) + l(4)*sin(theta(4))...
%          + l_tong*sin(theta(5));   
%     f(2) = Tx(6)/w*sinh(w/Tx(6)*x(6) + log(tan(alpha2) + sec(alpha2))) - Tx(6)/w*tan(alpha2)...
%         - L;
%     f(3) = Ty(6)/Tx(6) - sinh(w/Tx(6)*x(6) + log(tan(alpha2) + sec(alpha2)));
% else
%   
% end
f(1) = Tx(6)/w*cosh(w/Tx(6)*x(6) + log(tan(alpha2) + sec(alpha2))) - Tx(6)/w*sec(alpha2)...
    - y0 + l(1)*sin(theta(1)) + l(2)*sin(theta(2)) + l(3)*sin(theta(3)) + l(4)*sin(theta(4))...
     + l_tong*sin(theta(5));
f(2) = Tx(6)/w*sinh(w/Tx(6)*x(6) + log(tan(alpha2) + sec(alpha2))) - Tx(6)/w*tan(alpha2)...
    - L;
% f(3) = (Ty(6) - n*w)/Tx(6);
% f(3) = Ty(6)*cosh(w/Tx(6)*x(6) + log(tan(alpha2) + sec(alpha2))) - sqrt((Tx(6))^2 + (Ty(6))^2);
f(3) = Ty(6)/Tx(6) - sinh(w/Tx(6)*x(6) + log(tan(alpha2) + sec(alpha2)));  

%��������ϵͳͼ�����ֵ���
if xitong_figure == 1
    %1������ϵͳͼ��
    figure
    %ϵͳ����
    Y = @(X)Tx(6)/w*cosh(w/Tx(6)*X + log(tan(alpha2) + sec(alpha2))) - Tx(6)/w*sec(alpha2);
    X = linspace(x(6)-0.5, 0, 200);
    x = [x, X];
    y = [y, Y(X)];
    %����alpha2��if alpha2 < 0 then ......
    alpha2 = 90*alpha2/pi*2;%�Ƕ���
    if alpha2 < 0
        Ydao = @(X)sinh(w/Tx(6)*X + log(tan(alpha2) + sec(alpha2)));
        L_tuo = fzero(Ydao, [0+0.3, x0]);
        y(y<0) = 0;
    else
        L_tuo = 0;
    end
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
    title(['����',num2str(v1),'����ˮ���', num2str(abs(H-y0)), 'ʱ��ϵ��ϵͳ'])
    hold off
end

if xitong_save == 1 
    %2��������ֵ���
    Y = @(X)Tx(6)/w*cosh(w/Tx(6)*X + log(tan(alpha2) + sec(alpha2))) - Tx(6)/w*sec(alpha2);
    X = linspace(x(6)-0.5, 0, 200);
    x = [x, X];
    y = [y, Y(X)];
    %����alpha2��if alpha2 < 0 then ......
    alpha2 = 90*alpha2/pi*2;%�Ƕ���
    if alpha2 < 0
        L_tuo = fzero(Y, [0+0.3, x0]);
        y(y<0) = 0;
    else
        L_tuo = 0;
    end
    
    stat.h = abs(y0);
    stat.y0 = y0;
    stat.x0 = x0;
    stat.yn = y(end);
    stat.xn = x(end);
    stat.S = pi*x0^2;
    stat.alpha1 = 90 - 90*theta(5)/pi*2;%�Ƕ���
    stat.alpha2 = alpha2;%�Ƕ���
    stat.H = H;
    stat.v1 = v1;
    stat.v2 = v2;
    stat.m_qiu = m_qiu;
    stat.L = L;
    stat.L_tuo = L_tuo;
    stat.x = x;
    stat.y = y;
    stat.theta = theta;
    FILENAME = 'ϵͳ��Ϣ.mat';
    save(FILENAME, 'stat')
end

end















