%% �ֱ��Ͻ׶����
%����һ����������ֱ���˶�
clc;clear;close all;
Ve=2940;%�ȳ�
g=1.633;  %�����������ٶ�
h=2300;   %�ý׶ε��������
t=0;          %��ʼʱ��
T=0.1;       %ʱ�䲽��
%���ٵ����׶ε�ĩ״̬����Ϊ�ֱ��Ͻ׶εĳ�״̬��
Vy0=0.2208059;  %��ֱ���ٶ�
m0=1278.72898;%��ʼ����
count=0; %������
Y_res=Vy0*t;
TIME=h/Vy0;%������ʱ�䡣
Result=[];
%% ������ �ֽ��ٶȺͷֽ�λ��
M=m0;
consume=0;
while (Y_res<h )
count=count+1;
Y_res=Y_res+Vy0*T;
Time=count*T;
F=M*g;  %����
consume=consume+F/Ve*T;
M=M-F/Ve*T;
end
M%�ý׶ε�ĩ������
Time  %�˶�ʱ��
consume %��ȼ������





