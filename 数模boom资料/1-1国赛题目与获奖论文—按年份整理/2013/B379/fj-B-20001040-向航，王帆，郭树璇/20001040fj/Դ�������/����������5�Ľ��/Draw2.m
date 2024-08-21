[~,data]=xlsread('fujian7.xlsx');
col=[];
[m,n]=size(data);
for i=1:m
   row=[];
    for j=n:-1:1
       newData{i,n-j+1}=data{i,j};
       img=strcat(data{i,j},'.bmp');
       row=[row imread(img)];
    end
   col=[col;row];
end
imshow(col);