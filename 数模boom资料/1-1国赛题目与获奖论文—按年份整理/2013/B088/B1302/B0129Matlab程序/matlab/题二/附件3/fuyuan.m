clear all;clc;
load 'ht.mat';
load 'BMP3.mat';
t3=BMPfile3;

bm=t3;
for i=1:11
    for j=1:19
        AI([(1+(i-1)*180):(180*i)],[(1+(j-1)*72):(72*j)])=t3{ht(i,j),1};
    end
end
for i=1:11
    bmp{i,1}=AI(1+(i-1)*180:180*i,1:72*19);
end

d=[5 6 11 4 7 2 9 10 3 1 8];   %在这里人工干预调整行的排列顺序得到最终复原图
for i=1:11
    AI([(1+(i-1)*180):(180*i)],:)=bmp{d(i),1};
end



imwrite(AI,'hangtu.jpg','quality',100); 
imshow('hangtu.jpg')