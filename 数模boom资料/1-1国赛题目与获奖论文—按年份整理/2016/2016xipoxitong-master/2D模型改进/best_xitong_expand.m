%% ����bestpoint3_expand����bestx0, besty0����µ�ϵͳ��Ϣ��ϵͳͼ��
clc
clear
%����Ϊ12��24ʱ��ϵͳ��������⣡�������������н�0ʱ��ϵͳδ���úá�
H = 18;
v1 = 36;%���� m/s
v2 = 0;%ˮ�� m/s
m_qiu = 1200;%���������� kg
I = 2;
L = 22.05;

xitong_figure = 0;%�����ŵ�ʱ = 0������ϵͳʱ = 1��
xitong_save = 0;
bestxx = bestpoint3_expand(H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%�����ŵ�

xitong_figure = 1;
[~] = For2D_expand(bestxx, H, v1, v2, m_qiu, I, L, xitong_figure, xitong_save);%����ϵͳ
