%% �ֱ��Ͻ׶����
% ���Զ�  �����������壬��F���Ǻ�����
clc;clear;close all;tic;
Ve=2940;%�ȳ�
g=1.633;  %�����������ٶ�
h=2300;%�ý׶ε��������
t=0; %��ʼʱ��
T=0.1;   %ʱ�䲽��
lisan=7458:0.001:7460;
TIME=[];
M=[];
V_res=[];
Result=[];
F_res=[];
for j=lisan
%���ٵ����׶ε�ĩ״̬����Ϊ�ֱ��Ͻ׶εĳ�״̬��
V0=0.2208059;  %��ֱ���ٶ�
m0=1278.72898;%��ʼ����
count=0; %������
Y_res=V0*t;
Time=0;
%�����������������ʱ��T_max=52.9394466022199;
t1=45;%��������������ʱ��t1
h1=V0*t1+0.5*g*t1^2;%�������������h1
h2=h-h1;%�㶨�����ƶ�����h2
F=j;%�㶨����F
F_res=[F_res;F];
A0=g-F/(m0-F/Ve*t);%��ʼʱ�̼��ٶ�
V0=V0+g*t1; %�����γ��ٶ�
while (Y_res<h2 )
count=count+1;
Y=V0*T+0.5*A0*T^2;
V=V0+A0*T;
V0=V;
Y_res=Y_res+Y;
Time=count*T;
Ay=g-F/(m0-F/Ve*Time);
A0=Ay;
end
M=[M;m0-F/Ve*Time,F/Ve*Time];  %�ý׶ε�ĩ����
TIME=[TIME;Time+t1];  %���˶�ʱ��
V_res=[V_res;V];
end
Result=[Result;TIME,V_res,M,F_res];
toc;








