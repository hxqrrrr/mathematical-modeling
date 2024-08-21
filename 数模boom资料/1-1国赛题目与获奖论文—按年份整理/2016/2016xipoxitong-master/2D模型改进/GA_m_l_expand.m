function f = GA_m_l_expand(x, I, c1, c2, v1, v2, H, xitong_figure)
%�˺����ǵ��������m_qiu��L�Ż������Ŀ�꺯����������GA��fmincon������
%
%��Ϊm_qiu, L, I
%Ŀ�꣺��ˮ�����С�ζ�����͸�Ͱ�н���С
%

%%%%����%%%%
m_qiu = x(1);
L = x(2);

xitong_save = 0;
bestxx = bestpoint3_expand(H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%�����ŵ�
xitong_save = 1;
[~] = For2D_expand(bestxx, H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%����ϵͳ
load('ϵͳ��Ϣ.mat', 'stat')
alpha1 = stat.alpha1;
alpha2 = stat.alpha2;
h = stat.h;

%Ŀ��ֵ
f = h + c1*alpha1 + c2*pi*bestxx(2)^2;

disp('-----------------------------')
disp(['����������:', num2str(m_qiu)])
disp(['����:', num2str(L)])
disp(['����:', num2str(I)])
disp(['Ŀ��ֵf:', num2str(f)])
disp(['��ˮ���h:', num2str(h)])
disp(['x0:', num2str(stat.x0)])
disp(['alpha1:', num2str(alpha1)])
disp(['alpha2:', num2str(alpha2)])
disp('-----------------------------')
end









