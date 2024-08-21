function f = GA_m_l(x, I, c1, c2, v_wind, H, N, x0, y0_yn_figure, xitong_figure)
%�˺����ǵ��������m_qiu��L�Ż������Ŀ�꺯����������GA��fmincon������
%
%��Ϊm_qiu, L, I
%Ŀ�꣺��ˮ�����С�ζ�����͸�Ͱ�н���С
%

%%%%����%%%%
m_qiu = x(1);
L = x(2);

[besty0, bestx0] = bestpoint(H, N, x0, v_wind, m_qiu, I, L, y0_yn_figure);
[~, ~, ~, ~, stat] = For2D(besty0, bestx0, v_wind, m_qiu, I, L, xitong_figure);
alpha1 = stat.alpha1;
alpha2 = stat.alpha2;
h = abs(besty0);

%Ŀ��ֵ
f = h + c1*alpha1 + c2*pi*bestx0^2;

disp('-----------------------------')
disp(['����������:', num2str(m_qiu)])
disp(['����:', num2str(L)])
disp(['����:', num2str(I)])
disp(['Ŀ��ֵf:', num2str(f)])
disp(['��ˮ���h:', num2str(h)])
disp(['x0:', num2str(bestx0)])
disp(['alpha1:', num2str(alpha1)])
disp(['alpha2:', num2str(alpha2)])
disp('-----------------------------')
end









