clear all;clc;   %���븽��2ͼƬ����
cell{1,1}=imread('000.bmp');
cell{1,2}=imread('001.bmp');
cell{1,3}=imread('002.bmp');
cell{1,4}=imread('003.bmp');
cell{1,5}=imread('004.bmp');
cell{1,6}=imread('005.bmp');
cell{1,7}=imread('006.bmp');
cell{1,8}=imread('007.bmp');
cell{1,9}=imread('008.bmp');
cell{1,10}=imread('009.bmp');
cell{1,11}=imread('010.bmp');
cell{1,12}=imread('011.bmp');
cell{1,13}=imread('012.bmp');
cell{1,14}=imread('013.bmp');
cell{1,15}=imread('014.bmp');
cell{1,16}=imread('015.bmp');
cell{1,17}=imread('016.bmp');
cell{1,18}=imread('017.bmp');
cell{1,19}=imread('018.bmp');

for i=1:19
level=graythresh(cell{1,i});
cell1{1,i}=im2bw(cell{1,i},level); %Ϊ��������Ƚ���ͼ���ֵ������
end

for i=1:19
    for k=1:19
        xs(i,k)=0;
    for j=1:1980
        if(cell1{1,i}(j,72)==cell1{1,k}(j,1))  %�������ƶȾ���
            xs(i,k)=1+xs(i,k);
        end
    end
    end
end

%load 'ti1.mat';
for i=1:19
    xs(i,i)=0;
end

for i=1:19
da(i)=max(xs(i,:));
end
wei=find(da==max(da));
for i=1:19
    k=find(xs(i,[1:19])==da(i));  %����������ھ���
    lian(i,1)=i;                  %Ϊǰ��
    lian(i,2)=k;                  %Ϊ����
end
lian(wei,1)=0;
lian(wei,1)=0;
tou=lian(wei,2);
xu(1)=tou;
for i=1:18
    xu(i+1)=lian(xu(i),2); %  �����ȷ���������
end

%load 'xu.mat';
%��������ͼƬ��������xu��ԭͼƬ
for i=1:19
I(:,[72*(i-1)+1:72*i])=cell{1,xu(i)};
end

imwrite(I,'English.jpg','quality',100);
imshow('English.jpg')                  %���ͼƬ