[m,n]=size(tempSequence);
col=[];
for i=1:m
   row=[];
    for j=1:n
           if tempSequence(i,j)<=10
            img=strcat(strcat('00',int2str(tempSequence(i,j)-1)),'.bmp');
        elseif tempSequence(i,j)<=100
            img=strcat(strcat('0',int2str(tempSequence(i,j)-1)),'.bmp');    
        else
            img=strcat(int2str(tempSequence(i,j)-1),'.bmp');
           end
       row=[row imread(img)];
    end
   col=[col;row];
end
imshow(col);