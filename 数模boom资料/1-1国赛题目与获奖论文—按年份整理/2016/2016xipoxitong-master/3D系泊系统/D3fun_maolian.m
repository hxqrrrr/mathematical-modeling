function y = D3fun_maolian(x_point, T, theta, alpha, I)
%�˺����������ê����T��theta��alpha
%
%%%%%����˵��%%%%
% x���⡣T,theta��alpha
% T��Ti-1
% theta��


%%%%����%%%%
Tx = T*cos(theta)*cos(alpha);
Ty = T*cos(theta)*sin(alpha);
Tz = T*sin(theta);

%�ֹ���������
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
g = 9.8;
G_mao = II*m_II*g;%��λ��������

Ti = x_point(1);
thetai = x_point(2);
alphai = x_point(3);

Tix = Ti*cos(thetai)*cos(alphai);
Tiy = Ti*cos(thetai)*sin(alphai);
Tiz = Ti*sin(thetai);

y = [Tix - Tx;...
    Tiy - Ty;...
    Tiz + G_mao - Tz];
end