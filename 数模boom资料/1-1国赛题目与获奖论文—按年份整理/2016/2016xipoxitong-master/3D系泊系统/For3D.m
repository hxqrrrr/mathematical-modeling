function [z, y, x, theta, alpha, T, stat] = For3D(z0, y0, x0, v1, v2, m_qiu, I, L, beta, xitong_figure)
% �˺������ڸ���x0��y0��z0�����3Dϵ��ϵͳ��״̬����
%
%%%%����%%%%
% z0������z�����꣬|z0|=h�����У�hΪ��ˮ��ȡ�
% y0�����������ꡣ
% x0����������ꡣ
% v1�����١�
% v2��ˮ�١�
% m_qiu��������������
% I��ê���ͺš�1��2��3��4��5
% L��ê�����ȡ�
% beta��������ˮ���ļнǡ�
% xitongfigure���Ƿ����ϵͳͼ��logistic
%
%%%%���%%%%
% z��ϵ��ϵͳz�����ꡣ
% y��ϵ��ϵͳ�����ꡣ����
% x��ϵ��ϵͳ�����ꡣ
% theta��ϵ��ϵͳˮƽ��нǡ�
% alpha��ϵ��ϵͳx��������нǡ�
% T��ϵ��ϵͳ������
% stat��Ҫ���ϵ��ϵͳ��������������ˮ���h��������x0���ζ�����S����Ͱ��ֱ�н�alpha1��ê���׶�ˮƽ�н�alpha2������v_wind������������m��ϵͳ״̬yxthetaT��stats

%Ĭ������
if nargin < 10
    xitong_figure = 0;
end
if nargin < 9
    beta = pi/2;
end
if nargin < 8
    L = 22.05;%ê������ m
end
if nargin < 7
    I = 2;
end
if nargin < 6
    m_qiu = 500;
end
if nargin < 5
    v2 = 1.5;
end
if nargin < 4
    v1 = 12;
end
if nargin < 3
    x0 = -5;
end
if nargin < 2
    y0 = 20;
end
if nargin < 1
    disp('���������')
    return;
end

%%%%����%%%%
h = abs(z0);
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

%���ꡢ�ֹܷ���
fun = @(x_point)(D3fun_fubiao(x_point, beta, z0, v1, v2));
X0 = [14000, 0, pi/2];%T1��theta1��alpha1�ĳ�ʼ������
[x_solve, ~] = fsolve(fun, X0);

T(1) = x_solve(1);
theta(1) = x_solve(2);
alpha(1) = x_solve(3);
x(1) = x0;
y(1) = y0;
z(1) = z0;

for i = 1:4
    l = 1;%�ֹܳ���
    
    Ti = T(i);
    thetai = theta(i);
    alphai = alpha(i);
    fun = @(x_point)(D3fun_gangguan(x_point, Ti, thetai, alphai, beta, v2));
    X0 = [Ti, thetai, alphai];%T��theta��alpha�ĳ�ʼ������
    [x_solve, ~] = fsolve(fun, X0);
    
    T(i+1) = x_solve(1);
    theta(i+1) = x_solve(2);
    alpha(i+1) = x_solve(3);
    if alpha(i) < pi/2
        x(i+1) = x(i) + l*cos(theta(i))*cos(alpha(i));
    else
        x(i+1) = x(i) - l*cos(theta(i))*cos(alpha(i));
    end
    y(i+1) = y(i) - l*cos(theta(i))*sin(alpha(i));
    z(i+1) = z(i) - l*sin(theta(i));
end

%��Ͱ����
l_tong = 1;%��Ͱ�� m
T5 = T(5);
theta5 = theta(5);
alpha5 = alpha(5);
fun = @(x_point)(D3fun_gangtong(x_point, T5, theta5, alpha5, beta, v2, m_qiu));
X0 = [T5, theta5, alpha5];
[x_solve, ~] = fsolve(fun, X0);

T(6) = x_solve(1);
theta(6) = x_solve(2);
alpha(6) = x_solve(3);
if alpha(5) < pi/2
    x(6) = x(5) + l_tong*cos(theta(5))*cos(alpha(5));
else
    x(6) = x(5) - l_tong*cos(theta(5))*cos(alpha(5));
end
y(6) = y(5) - l_tong*cos(theta(5))*sin(alpha(5));
z(6) = z(5) - l_tong*sin(theta(5));

%ê���߷���
L_tuo = 0;%ê����β����
for i = 6 : 6+n-1
    if  theta(i) - 0 >0.001
        Ti = T(i);
        thetai = theta(i);
        alphai = alpha(i);
        fun = @(x_point)(D3fun_maolian(x_point, Ti, thetai, alphai, I));
        X0 = [Ti, thetai, alphai];%T��theta��alpha�ĳ�ʼ������
        [x_solve, ~] = fsolve(fun, X0);

        T(i+1) = x_solve(1);
        theta(i+1) = x_solve(2);
        alpha(i+1) = x_solve(3);
        if alpha(i) < pi/2
            x(i+1) = x(i) + II*cos(theta(i))*cos(alpha(i));
        else
            x(i+1) = x(i) - II*cos(theta(i))*cos(alpha(i));
        end
        y(i+1) = y(i) - II*cos(theta(i))*sin(alpha(i));
        z(i+1) = z(i) - II*sin(theta(i));
    else 
        T(i+1) = 0;
        theta(i+1) = 0;
        alpha(i+1) = alpha(i);
        
        z(i+1) = z(i);
        y(i+1) = y(i) - II*sin(alpha(i));
        if alpha(i) < pi/2
            x(i+1) = x(i) + II*cos(alpha(i));
        else
            x(i+1) = x(i) - II*cos(alpha(i));
        end
        L_tuo = L_tuo+II;
    end
end


%�������[z, y, x, theta, alpha, T, stat]
if nargout == 7
    z = z;
    y = y;
    x = x;
    theta = theta;
    alpha = alpha;
    T= T; 
    % stat��Ҫ���ϵ��ϵͳ��������������ˮ���h��������x0���ζ�����S����Ͱ��ֱ�н�alpha1��ê���׶�ˮƽ�н�alpha2������v_wind������������m��ϵͳ״̬yxthetaT
    stat.h = abs(z0);
    stat.z0 = z0;
    stat.y0 = y0;
    stat.x0 = x0;
    stat.zn = z(end);
    stat.yn = y(end);
    stat.xn = x(end);
    stat.S = pi*(x0^2 + y0^2);
    stat.alpha1 = 90 - 90*theta(5)/pi*2;%�Ƕ���
    stat.alpha2 = 90*theta(end-1)/pi*2;%�Ƕ���
    stat.v1 = v1;
    stat.v2 = v2;
    stat.m_qiu = m_qiu;
    stat.beta = beta;
    stat.L_tuo = L_tuo;
    stat.x = x;
    stat.y = y;
    stat.z = z;
    stat.theta = theta;
    stat.alpha = alpha;
    stat.T = T;
end

if xitong_figure == 1
    figure
    %ϵͳ����
    plot3(x(6:end), y(6:end), z(6:end), '-', 'color', rand(1, 3))
    hold on
    plot3(x(1:5), y(1:5), z(1:5), '-*y', 'LineWidth', 2)
    plot3(x(5:6), y(5:6), z(5:6), '-r', 'LineWidth', 12)
    
    %��Ư
    R = 1;%�뾶
    h0 = 2;%Բ���߶�
    mm = 100;%�ָ��ߵ�����
    [xx , yy, zz] = cylinder(R, mm);%������(0,0)ΪԲ�ģ��߶�Ϊ[0,1]���뾶ΪR��Բ��
    xx = xx + x0;%ƽ��x��
    yy = yy + y0;%ƽ��y�ᣬ��Ϊ(a,b)Ϊ��Բ��Բ��
    zz = h0*zz + z0;%ƽ��z�ᣬ�߶ȷŴ�h��
    mesh(xx, yy, zz)%���»�ͼ
    
    fubiao_r = 1;%��ӷⶥ����
    t = linspace(0, pi, 25);
    p = linspace(0, 2*pi, 25);
    x1 = fubiao_r*cos(p) + x0;
    y1 = fubiao_r*sin(p) + y0;
    z1 = z0*ones(size(x1));
    fill3(x1, y1, z1, 'b')
    z2 = (h0 + z0)*ones(size(x1));
    fill3(x1, y1, z2, 'b')
    
    %������
    plot3([x(6), x(6)], [y(6), y(6)], [z(6), z(6)-1.5], '-')
    qiu_r = 0.25;
    [the, phi] = meshgrid(t, p);
    xxx = qiu_r*sin(the).*sin(phi);
    yyy = qiu_r*sin(the).*cos(phi);
    zzz = qiu_r*cos(the);
%     [xxx, yyy, zzz] = sphere;
    mesh(xxx + x(6), yyy + y(6), zzz + z(6)-1.5)%centered at (x6,y6,z6-1.5)
%     alpha(0.3)%����ͼ�ε�͸���ȣ�ȡֵ0~1 

    %ê
    boxplot3(x(end) - 0.5, y(end), z(end), 1, 1, 1)
    
    %ע��
    alpha1 = 90 - 90*theta(5)/pi*2;%�Ƕ���
    alpha2 = 90*theta(end-1)/pi*2;%�Ƕ���
    text(x0, y0-2, z0-1, [num2str(x0),',',num2str(y0),',',num2str(z0)])
    text(x(6), y(6)-3, z(6)-2, ['����������:', num2str(m_qiu)])
    text(x(6), y(6)-0.5, z(6), ['��Ͱ���:', num2str(alpha1)])
    text(x(end), y(end)+0.5, z(end)+1.5, ['ê���н�:', num2str(alpha2)])
    if L_tuo ~= 0
        text(x(end), y(end)+0.5, z(end)+0.5, ['��β����:', num2str(L_tuo)])
    end
    
    %ͼ������
%     box off
    view(3)%�����ӽ�
    axis equal
    xlabel('x�᷽��')
    ylabel('��������')
    zlabel('��ֱ����')
    title(['����',num2str(v1),'����ˮ����', num2str(abs(z0)), 'ʱ��ϵ��ϵͳ'])
    hold off
end


%%%%����Բ����%%%%
% % Sample values
% h0 = 2; % height
% ra = 1; % radius
% % Create constant vectors
% tht = linspace(0, 2*pi, 100);
% z = linspace(0, h0, 20);
% % Create cylinder
% xa = repmat(ra*cos(tht),20,1); ya = repmat(ra*sin(tht),20,1);
% za = repmat(z',1,100);
% % To close the ends
% X = [xa*0; flipud(xa); (xa(1,:))*0]; 
% Y = [ya*0; flipud(ya); (ya(1,:))*0];
% Z = [za; flipud(za); za(1,:)];
% % Draw cylinder
% [TRI,v] = surf2patch(X,Y,Z,'triangle');
% patch('Vertices',v,'Faces',TRI,'facecolor',[0.5 0.8 0.8],'facealpha',0.8);
% view(3);
% grid on; 
% axis square; 
% title('Cylinder','FontSize',12)


end







