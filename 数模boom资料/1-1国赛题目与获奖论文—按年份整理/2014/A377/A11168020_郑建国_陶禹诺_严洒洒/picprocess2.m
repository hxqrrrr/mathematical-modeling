%% �ڶ������ָ߳�ͼ�Ĵ���
clc;clear;close all;tic;
z=imread('����4 ������100m�������ָ߳�ͼ.tif');
%z=double(z);
% x=1:length(K);
% y=x;
% [X,Y]=meshgrid(x,y);
% mesh(X,Y,double(K));
% colormap(gray);
% colorbar;
% imshow(K);
%% ��������
temp=z(51:950,51:950);%ת��Ϊ�ɾ��ֵ�900X900�Ź������
for i=1:9
    switch i
         case   {1,2,3}
     G{i}=temp(1:300,1+(i-1)*300:i*300);
         case   {4,5,6}
     G{i}=temp(301:600,1+(i-4)*300:(i-3)*300);
         case   {7,8,9}
     G{i}=temp(601:end,1+(i-7)*300:(i-6)*300);
    end
end
for i=1:9
    b=i;
    a=330+i;
   subplot(a);
   imshow(G{1,i});
end
%% 9������ĸ���ͳ��������
MEAN=[];  %�߳̾�ֵ
JICHA=[];   %�̼߳���
STD=[];      %�̱߳�׼��
XD=[];        %�����ֵ����������ֵ�ġ���Ը̡߳�
ZT=mean(temp(:));%�����ֵ
for i=1:9
    TEMP=G{1,i};
    TEMP=double(TEMP(:));
    MEAN=[MEAN,mean(TEMP)];
    MAX=max(TEMP);
    MIN=min(TEMP);
    JICHA=[JICHA,MAX-MIN];
    STD=[STD,std(TEMP)];
    XD=[XD,abs(MEAN(i)-ZT)/ZT];
end
result=[MEAN;JICHA;STD;XD];

%% STD XD �Ĺ�һ��
m1=max(STD);
m2=min(STD);

m3=max(XD);
m4=min(XD);

STD2=(STD-m2)/(m1-m2);
XD2=(XD-m4)/(m3-m4);
RESULT=[MEAN;JICHA;STD2;XD2;STD2+XD2];

%% �ȸ���ͼ�Ļ���
% figure;
% %z=double(z);
% x=1:length(z);
% y=x;
% [X2,Y2]=meshgrid(x,y);
% subplot(121);
% contour(X2,Y2,z);
% title('������100m���ĵȸ���ͼ','FontSize',14);
% colormap(gray);
% z1=G{1};
% x=1:length(z1);
% y=x;
% [X2,Y2]=meshgrid(x,y);
% subplot(122);
% contour(X2,Y2,z1);
% colormap(gray);colorbar;
% title('1������ȸ���ͼ','FontSize',14);
toc;