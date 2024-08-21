function [ picType,picDistance ] = pictureAttribute( data )
%pictureType is to determine the picture Attribute.
%black to white is 0,white to black is 1.
%flag 1 is black,flag 0 is white.

[M,~]=size(data);

if length(find(data(1,:)==0))~=0
    flag=1;
else
    flag=0;
end

for i=1:M
    
    if flag==1
        if length(find(data(i,:)==0))==0
           picType(1,1)=0;
           break;
        end
    else
        if length(find(data(i,:)==0))~=0
           picType(1,1)=1;
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
           break;
        end
    else
        if length(find(data(j,:)==0))~=0
           picType(1,2)=1;
           break;
        end
    end    
end

count=1;

for i=1:72

    if length(find(data(1,:)==0))==0
        flag=1;

    else
        flag=0; 
    end
    
    for j=1:180
    
        if flag==0; 
            if data(j,i)==1
               distance(count)=j;
               count=count+1;
               break; 
            end
        else
            if data(j,i)==0
               distance(count)=j;
               count=count+1;
               break; 
            end
        end
    end
    
end
a=mode(distance);
clear distance;

count=1;

for i=1:72

    if length(find(data(180,:)==0))==0
        flag=1;
      
    else
        flag=0; 
        
    end
    
    for j=180:-1:1
    
        if flag==0; 
            if data(j,i)==1
               distance(count)=j;
               count=count+1;
               break; 
            end
        else
            if data(j,i)==0
               distance(count)=j;
               count=count+1;
               break; 
            end
        end
    end
    
end
b=mode(distance);




picDistance=[a b];

end