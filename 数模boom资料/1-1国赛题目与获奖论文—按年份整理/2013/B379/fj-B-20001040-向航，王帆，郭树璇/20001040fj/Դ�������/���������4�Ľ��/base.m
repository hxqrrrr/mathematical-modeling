function [ temp ] = base( image )
%base is to find the baseline of each picture.
%   image is the 180*72 picture.

data=im2bw(image,graythresh(image));

[row,col]=size(data);

for i=1:row
    temp(i,1)=sum(data(i,:));    
end


end