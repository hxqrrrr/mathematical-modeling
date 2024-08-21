


[~,data]=xlsread('fujian5.xlsx');
col=[];
[m,n]=size(data);
for i=1:m
   row=[];
    for j=1:n
           
       img=strcat(data{i,j},'.bmp');
       row=[row imread(img)];
    end
   col=[col;row];
end
imshow(col);