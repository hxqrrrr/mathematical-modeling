function [ picType,picDistance ] = pictureAttribute( data )
%pictureType is to determine the picture Attribute.
%black to white is 0,white to black is 1.
%flag 1 is black,flag 0 is white.

[M,~]=size(data);
picType=[];

if length(find(data(1,:)==0))~=0
    flag=1;
else
    flag=0;
end

for i=1:M
    
    if flag==1
        if length(find(data(i,:)==0))==0
           picType(1,1)=0;
           picDistance(1,1)=i-1;
           break;
        end
    else
        if length(find(data(i,:)==0))~=0
           picType(1,1)=1;
           picDistance(1,1)=i-1;
           break;
        end
    end
    
end

if length(find(data(M,:)==0))~=0
    flag=1;
else
    flag=0;
end

for j=M:-1:1
   
    if flag==1
        if length(find(data(j,:)==0))==0
           picType(1,2)=0;
           picDistance(1,2)=M-j;
           break;
        end
    else
        if length(find(data(j,:)==0))~=0
           picType(1,2)=1;
           picDistance(1,2)=M-j;
           break;
        end
    end    
end

end