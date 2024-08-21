%% ����v1��ϵͳ״̬��Ӱ��
clc
clear
% �����Բ���v1
v1 = 6:6:36;
%��������
H = 18;
v2 = 1.5;
m_qiu = 1200;
I = 2;
L = 22.05;
xitong_figure = 0;%�����ŵ�ʱ = 0������ϵͳʱ = 1��

%%%%����%%%%
figure(1)
for i = 1:length(v1)
    A{i} = ['����', num2str(v1(i))];
    xitong_save = 0;
    bestxx = bestpoint3_expand(H, v1(i), v2, m_qiu, I, L, xitong_figure, xitong_save);%�����ŵ�

    xitong_save = 1;
    [~]= For2D_expand(bestxx, H, v1(i), v2, m_qiu, I, L, xitong_figure, xitong_save);%����ϵͳ
    load('ϵͳ��Ϣ.mat', 'stat')
    x(:, i) = stat.x;
    y(:, i) = stat.y;
    plot(x(:, i), y(:, i), '-', 'color', rand(3, 1))
    hold on
end
hold off
legend(A, 'location', 'best')
xlabel('����')
ylabel('ϵͳ״̬')
title('����v1��ϵ��ϵͳ��Ӱ��')

%% ˮ��v2��ϵͳ״̬��Ӱ��
clc
clear
% �����Բ���v1
v2 = -1.5:0.5:1.5;
%��������
H = 18;
v1 = 36;
m_qiu = 1200;
I = 2;
L = 22.05;
xitong_figure = 0;
%%%%����%%%%
figure(2)
for i = 1:length(v2)
    A{i} = ['ˮ��', num2str(v2(i))];
    xitong_save = 0;%�����ŵ�ʱ = 0������ϵͳʱ = 1��
    bestxx = bestpoint3_expand(H, v1, v2(i), m_qiu, I, L, xitong_figure, xitong_save);%�����ŵ�

    xitong_save = 1;
    [~]= For2D_expand(bestxx, H, v1, v2(i), m_qiu, I, L, xitong_figure, xitong_save);%����ϵͳ
    load('ϵͳ��Ϣ.mat', 'stat')
    x(:, i) = stat.x;
    y(:, i) = stat.y;
    plot(x(:, i), y(:, i), '-', 'color', rand(3, 1))
    hold on
end
hold off
legend(A, 'location', 'best')
xlabel('����')
ylabel('ϵͳ״̬')
title('ˮ��v2��ϵ��ϵͳ��Ӱ��')

%% ��ˮ���H��ϵͳ״̬��Ӱ��
clc
clear
% �����Բ���v1
H = 16:20;
%��������
v1 = 36;
v2 = 1.5;
m_qiu = 1200;
I = 2;
L = 22.05;
xitong_figure = 0;

%%%%����%%%%
figure(3)
for i = 1:length(H)
    A{i} = ['ˮ��', num2str(H(i))];
    xitong_save = 0;%�����ŵ�ʱ = 0������ϵͳʱ = 1��
    bestxx = bestpoint3_expand(H(i), v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%�����ŵ�

    xitong_save = 1;
    [~]= For2D_expand(bestxx, H(i), v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%����ϵͳ
    load('ϵͳ��Ϣ.mat', 'stat')
    x(:, i) = stat.x;
    y(:, i) = stat.y;
    plot(x(:, i), y(:, i), '-', 'color', rand(3, 1))
    hold on
end
hold off
legend(A, 'location', 'best')
xlabel('����')
ylabel('ϵͳ״̬')
title('ˮ��H��ϵ��ϵͳ��Ӱ��')




































