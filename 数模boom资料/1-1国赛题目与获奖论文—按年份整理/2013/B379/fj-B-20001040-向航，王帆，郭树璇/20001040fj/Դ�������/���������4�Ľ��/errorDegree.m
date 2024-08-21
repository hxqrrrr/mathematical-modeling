function [ degree1,degree2,degree3 ] = errorDegree( A,B,tao )
%errorDegree is to calculate the error degree of two picture.
%   A is the left Matrix,B is the right Matrix.

n=length(A);
tempError=zeros;
Error=zeros(1,180);
for i=3:(n-2)
    
    if i<=60
    temp1=0.7*(A(i,2)-B(i,1));
    temp2=0.1*(A(i-1,2)-B(i-1,1));
    temp3=0.1*(A(i+1,2)-B(i+1,1));
    temp4=0.05*(A(i-2,2)-B(i-2,1));
    temp5=0.05*(A(i+2,2)-B(i+2,1));
    tempError(i)=abs(temp1+temp2+temp3+temp4+temp5);
    if tempError(i)>tao
        Error(i)=1;
    else
        Error(i)=0;
    end
    continue;
    
    
    elseif i<=119 && i>60
    temp1=0.7*(A(i,2)-B(i,1));
    temp2=0.1*(A(i-1,2)-B(i-1,1));
    temp3=0.1*(A(i+1,2)-B(i+1,1));
    temp4=0.05*(A(i-2,2)-B(i-2,1));
    temp5=0.05*(A(i+2,2)-B(i+2,1));
    tempError(i)=abs(temp1+temp2+temp3+temp4+temp5);
    if tempError(i)>tao
        Error(i)=1;
    else
        Error(i)=0; 
    end
    continue;
   
    else
    temp1=0.7*(A(i,2)-B(i,1));
    temp2=0.1*(A(i-1,2)-B(i-1,1));
    temp3=0.1*(A(i+1,2)-B(i+1,1));
    temp4=0.05*(A(i-2,2)-B(i-2,1));
    temp5=0.05*(A(i+2,2)-B(i+2,1));
    tempError(i)=abs(temp1+temp2+temp3+temp4+temp5);
    if tempError(i)>tao
        Error(i)=1;
    else
        Error(i)=0;
    end
    continue;
        
    end
    
end
degree1=sum(Error(1,1:60));
degree2=sum(Error(1,61:119));
degree3=sum(Error(1,120:180));
end

