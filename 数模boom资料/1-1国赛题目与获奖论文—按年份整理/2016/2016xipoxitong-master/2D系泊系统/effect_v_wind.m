%% ���ٶ�ϵͳ״̬��Ӱ��
clc
clear
% �����Բ���v_wind
v_wind = 10:5:40;
%��������
H = 18;
N = 1000;
x0 = 20;
m_qiu = 1200;
I = 2;
L = 22.05;
y0_yn_figure = 0;
xitong_figure = 0;
%%%%����%%%%
figure(1)
for i = 1:length(v_wind)
    A{i} = ['����', num2str(v_wind(i))];
    [besty0, bestx0] = bestpoint(H, N, x0, v_wind(i), m_qiu, I, L, y0_yn_figure);
    y0 = besty0;
    x0 = bestx0;
    [y(:, i), x(:, i), theta(:, i), T(:, i), stat(i)] = For2D(y0, x0, v_wind(i), m_qiu, I, L, xitong_figure);
    plot(x(:, i), y(:, i), '-', 'color', rand(3, 1))
    hold on
end
legend(A, 'location', 'best')
xlabel('����')
ylabel('ϵͳ״̬')
title('���ٶ�ϵ��ϵͳ��Ӱ��')
title('')
figure(2)
for i = 1:length(v_wind)
    plot(theta(:, i), '-', 'color', rand(3, 1))
    hold on
end
legend(A, 'location', 'best')
xlabel('����')
ylabel('ϵͳ������ˮƽ�н�')
title('���ٶ�ϵͳˮƽ�нǵ�Ӱ��')