%% �����ٽ׶ε���ֵ������⣨���ڵ�һ�ʣ�
clc;clear;close all;
g=1.633;  %�����������ٶ�
m0=2.4*10^3;%���ǳ�ʼ����
Ve=2940; %�ȳ�
theta=9.654*pi/180;%���ٶ���ˮƽ����ļн�
F=7500;   %����
V0=1692.464; %���յ���ٶ�
t=0;          %��ʼʱ��
T=0.1;       %ʱ�䲽��
Vx0=V0*cos(-theta); %ˮƽ���ٶ�
Vy0=V0*sin(-theta); %��ֱ���ٶ�
Ay0=g-F*sin(-theta)/(m0-F/Ve*t);%��ֱ�����ٶ�
Ax0=-F*cos(-theta)/(m0-F/Ve*t);%ˮƽ�����ٶ�
count=0;
X_res=Vx0*t+0.5*Ax0*t^2;
Y_res=Vy0*t+0.5*Ax0*t^2;
Result=[];
h=12000;%�����ٽ׶ε��������
%% ������ �ֽ��ٶȺͷֽ�λ��
while (Y_res<h )
count=count+1;
Vx=Vx0+Ax0*T;
Vy=Vy0+Ay0*T;
V_res=sqrt(Vx^2+Vy^2);
Vx0=Vx;
Vy0=Vy;
X=Vx0*T+0.5*Ax0*T^2;
Y=Vy0*T+0.5*Ay0*T^2;
X_res=X_res+X;
Y_res=Y_res+Y;
 Time=count*T;
 Result=[Result;X_res,Y_res,V_res,Time];
SIN=Vy/sqrt(Vy^2+Vx^2);
COS=Vx/sqrt(Vy^2+Vx^2);
Ay=g-F*SIN/(m0-F/Ve*Time);
Ax=-F*COS/(m0-F/Ve*Time);
Ax0=Ax;
Ay0=Ay;
end
M=m0-F/Ve*Time;%�ý׶ε�ĩ������
X_res  %ˮƽλ��
 Time=count*T  %�˶�ʱ��
 V_res=sqrt(Vx^2+Vy^2) %���ٶ�
 jiaodu=atan(Vy/Vx)*180/pi %ĩ�ٶȽǶȡ�
 F/Ve*Time%ȼ��������
%% ��ͼ
plot(Result(:,1),15000-Result(:,2),'k','LineWidth',2);
title('�����ٶ��϶������˶��켣ͼ','FontSize',15);
xlabel('X��/(m)');
ylabel('Y��/(m)');
%% ��γ�ȵĽ��Ƽ���
R=1737.013; %ƽ���뾶
weidu=44.12-X_res/((2*pi*R)/4/90)/1000  %���µ�γ��
pi*R/180;  % 30.3165
