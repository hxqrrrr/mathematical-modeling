%% ���ٵ����׶ε���ֵ������⡣
clc;clear;close all;
Ve=2940;%�ȳ�
g=1.633;  %�����������ٶ�
h=600;%�ý׶ε��������
t=0; %��ʼʱ��
T=0.1;   %ʱ�䲽��
M_temp=[];
V_temp=[];
shijian=[];
X_temp=[];
lisan=1500:100:7500;
%lisan=5000:5100;
for i = lisan
F=i;  %����
%�����ٽ׶ε�ĩ״̬����Ϊ���ٵ����׶εĳ�״̬��
theta=55.6708*pi/180;%���ٶ���ˮƽ��ļн�
Vx0=32.23327;%ˮƽ���ٶ�
Vy0=47.2005;  %��ֱ���ٶ�
m0=1325.255;%��ʼ����
Ay0=g-F*sin(theta)/(m0-F/Ve*t);%��ֱ�����ٶ�
Ax0=-F*cos(theta)/(m0-F/Ve*t);%ˮƽ�����ٶ�
count=0; %������
X_res=Vx0*t+0.5*Ax0*t^2;
Y_res=Vy0*t+0.5*Ax0*t^2;
Result=[];

%% ������ �ֽ��ٶȺͷֽ�λ��
while (Y_res<h )
count=count+1;
Vx=Vx0+Ax0*T;
Vy=Vy0+Ay0*T;
Vx0=Vx;
Vy0=Vy;
X=Vx0*T+0.5*Ax0*T^2;
Y=Vy0*T+0.5*Ay0*T^2;
X_res=X_res+X;
Y_res=Y_res+Y;
 Time=count*T;
SIN=Vy/sqrt(Vy^2+Vx^2);
COS=Vx/sqrt(Vy^2+Vx^2);
Ay=g-F*SIN/(m0-F/Ve*Time);
Ax=-F*COS/(m0-F/Ve*Time);
Ax0=Ax;
Ay0=Ay;
end
M=m0-F/Ve*Time;%�ý׶ε�ĩ������
X_res; %ˮƽλ��
 Time=count*T;  %�˶�ʱ��
 V_res=sqrt(Vx^2+Vy^2);%���ٶ�
 jiaodu=atan(Vy/Vx)*180/pi; %ĩ�ٶȽǶ�
 %Vx  %ˮƽ�ٶ�
M_temp=[M_temp;F/Ve*Time];
V_temp=[V_temp;V_res,Vx];
shijian=[shijian;Time];%��¼����ʱ��
X_temp=[X_temp;X_res];
end
Answer=[lisan',M_temp,V_temp,shijian,X_temp];%����ܽ�������
subplot(221);
plot(lisan,M_temp,'LineWidth',2);
title('ȼ�����������������仯ͼ','FontSize',14);
xlabel('F_�ƣ�N��','FontSize',12);
ylabel('ȼ����������(ǧ��)');

subplot(222);
plot(lisan,X_temp,'LineWidth',2);
title('ˮƽλ�ƹ��������仯ͼ','FontSize',14);
xlabel('F_�ƣ�N��','FontSize',12);
ylabel('λ��(��)');

subplot(223);
plot(lisan,V_temp(:,2),'LineWidth',2);
title('ˮƽĩ�ٶȹ��������仯ͼ','FontSize',14);
xlabel('F_�ƣ�N��','FontSize',12);
ylabel('�ٶ�(��/��)');

subplot(224);
plot(lisan,shijian,'LineWidth',2);
title('����ʱ����������仯ͼ','FontSize',14);
xlabel('F_�ƣ�N��','FontSize',12);
ylabel('ʱ��(��)');


%% ���ݱ���д�롣
M_temp=[];
V_temp=[];
shijian=[];
X_temp=[];
lisan=1500:7500;
for i = lisan
F=i;  %����
%�����ٽ׶ε�ĩ״̬����Ϊ���ٵ����׶εĳ�״̬��
theta=55.6708*pi/180;%���ٶ���ˮƽ��ļн�
Vx0=32.23327;%ˮƽ���ٶ�
Vy0=47.2005;  %��ֱ���ٶ�
m0=1325.255;%��ʼ����
Ay0=g-F*sin(theta)/(m0-F/Ve*t);%��ֱ�����ٶ�
Ax0=-F*cos(theta)/(m0-F/Ve*t);%ˮƽ�����ٶ�
count=0; %������
X_res=Vx0*t+0.5*Ax0*t^2;
Y_res=Vy0*t+0.5*Ax0*t^2;
Result=[];

%% ������ �ֽ��ٶȺͷֽ�λ��
while (Y_res<h )
count=count+1;
Vx=Vx0+Ax0*T;
Vy=Vy0+Ay0*T;
Vx0=Vx;
Vy0=Vy;
X=Vx0*T+0.5*Ax0*T^2;
Y=Vy0*T+0.5*Ay0*T^2;
X_res=X_res+X;
Y_res=Y_res+Y;
 Time=count*T;
SIN=Vy/sqrt(Vy^2+Vx^2);
COS=Vx/sqrt(Vy^2+Vx^2);
Ay=g-F*SIN/(m0-F/Ve*Time);
Ax=-F*COS/(m0-F/Ve*Time);
Ax0=Ax;
Ay0=Ay;
end
M=m0-F/Ve*Time;%�ý׶ε�ĩ������
X_res; %ˮƽλ��
 Time=count*T;  %�˶�ʱ��
 V_res=sqrt(Vx^2+Vy^2);%���ٶ�
 jiaodu=atan(Vy/Vx)*180/pi; %ĩ�ٶȽǶ�
 %Vx  %ˮƽ�ٶ�
M_temp=[M_temp;F/Ve*Time];
V_temp=[V_temp;V_res,Vx];
shijian=[shijian;Time];%��¼����ʱ��
X_temp=[X_temp;X_res];
end
Answer=[lisan',M_temp,V_temp,shijian,X_temp];
%xlswrite('���ٵ����׶θ���������.xls',Answer);
%��һ��������ֵ
%�ڶ�����ȼ��������
%�������ǿ��ٵ�����ĩ�ٶ�
%�������ǿ��ٵ����ε�ˮƽĩ�ٶ�
%������������ʱ��
%��������ˮƽλ��

%%  ���Ź켣ͼ�Ļ���
figure;
M_temp=[];
V_temp=[];
shijian=[];
F=5085;  %����
%�����ٽ׶ε�ĩ״̬����Ϊ���ٵ����׶εĳ�״̬��
theta=55.6708*pi/180;%���ٶ���ˮƽ��ļн�
Vx0=32.23327;%ˮƽ���ٶ�
Vy0=47.2005;  %��ֱ���ٶ�
m0=1325.255; %��ʼ����
Ay0=g-F*sin(theta)/(m0-F/Ve*t);%��ֱ�����ٶ�
Ax0=-F*cos(theta)/(m0-F/Ve*t); %ˮƽ�����ٶ�
count=0; %������
X_res=Vx0*t+0.5*Ax0*t^2;
Y_res=Vy0*t+0.5*Ax0*t^2;
Result=[];
G=[];
%% ������ �ֽ��ٶȺͷֽ�λ��
while (Y_res<h )
count=count+1;
Vx=Vx0+Ax0*T;
Vy=Vy0+Ay0*T;
Vx0=Vx;
Vy0=Vy;
X=Vx0*T+0.5*Ax0*T^2;
Y=Vy0*T+0.5*Ay0*T^2;
X_res=X_res+X;
Y_res=Y_res+Y;
 Time=count*T;
G=[G;X_res,Y_res,V_res,Time];
SIN=Vy/sqrt(Vy^2+Vx^2);
COS=Vx/sqrt(Vy^2+Vx^2);
Ay=g-F*SIN/(m0-F/Ve*Time);
Ax=-F*COS/(m0-F/Ve*Time);
Ax0=Ax;
Ay0=Ay;
end
plot(G(:,1),3000-G(:,2),'k','LineWidth',2);
title('���ٵ������˶��켣','FontSize',15);
xlabel('X��/(m)');
ylabel('Y��/(m)');







