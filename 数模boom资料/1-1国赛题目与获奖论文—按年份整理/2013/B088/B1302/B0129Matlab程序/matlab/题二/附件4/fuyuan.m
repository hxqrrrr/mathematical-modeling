clear all;clc;
ht=load( 'ht.txt');
load 't4.mat';

bm=t4;
for i=1:11
    for j=1:19
        AI([(1+(i-1)*180):(180*i)],[(1+(j-1)*72):(72*j)])=bm{ht(i,j),1};
    end
end
for i=1:11
    bmp{i,1}=AI(1+(i-1)*180:180*i,1:72*19);
end

d=[1 2 3 4 5 6 7 8 9 10 11];  %��Ϊͨ���˹���Ԥ�����е�����õ�����ȷ������
for i=1:11
    AI([(1+(i-1)*180):(180*i)],:)=bmp{d(i),1};
end



imwrite(AI,'hangtu.jpg','quality',100); 
imshow('hangtu.jpg')
