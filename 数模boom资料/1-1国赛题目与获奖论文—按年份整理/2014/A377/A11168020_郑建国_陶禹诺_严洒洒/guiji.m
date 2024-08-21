%% ��������½���̵��˶��켣
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
GJ=[Result(:,1),15000-Result(:,2)];
%%
clc;close all;
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
GJ2=[G(:,1)+GJ(end,1),3000-G(:,2)];
flag1=[377092.173014911,2999.65005894814];%�����ٶν�����
flag2=[377381.089941517,2399.99226916603];%���ٵ����ν�����
GJ3=[GJ;GJ2];
%plot(GJ3(:,1),GJ3(:,2));
hold on;
%plot(flag1(1,1),flag1(1,2),'o','MarkerSize',20);
%plot(flag2(1,1),flag2(1,2),'o','MarkerSize',20);
ttm=2399:-1:100;
CUBI=[zeros(2300,1)+flag2(1,1),ttm'];
GJ4=[GJ3;CUBI];
%plot(GJ4(:,1),GJ4(:,2));
flag3=[377381.089941517,100];%�ֱ��϶ν�����
%%
h=70;
aaa=1.98;
T=sqrt(2*h/g);
t1=0.5*T;
t2=t1;
x3=[];
y3=[];
x33=[];

for i=0:0.01:t1
x3=[x3;0.5*1.98*i^2];
end
temp5=x3(end,1);
vvv=aaa*t1;

for i=0:0.01:t2
x33=[x33;temp5+vvv*i-0.5*aaa*i^2];
end
X3=[x3;x33];
for i=0:0.01:T
y3=[y3;100-0.5*g*i^2];
end
x5=[GJ4(end,1)+X3];
GJ5=[GJ4;x5,y3];
%plot(GJ5(:,1),GJ5(:,2));
ttn=29:-1:0;
TT5=[zeros(30,1)+GJ5(end,1),ttn'];
GJ6=[GJ5;TT5];
plot(GJ6(:,1),GJ6(:,2),'LineWidth',2);
title('��4�����̵Ĺ켣ͼ','FontSize',14);
xlabel('ˮƽλ��/��');
ylabel('�߶�/��');
%axis([377092.173014911,377500,0 3000])