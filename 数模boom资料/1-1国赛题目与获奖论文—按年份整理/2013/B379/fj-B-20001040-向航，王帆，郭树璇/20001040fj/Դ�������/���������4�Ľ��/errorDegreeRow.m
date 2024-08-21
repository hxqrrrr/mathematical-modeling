function [ degree] = errorDegreeRow( A,B,tao )
%errorDegree is to calculate the error degree of two picture.
%   A is the left Matrix,B is the right Matrix.
A=int16(A);
B=int16(B);
n=length(A);
tempError=zeros;
Error=zeros(1,1368);
for i=3:(n-2)
    temp1=0.7*(A(2,i)-B(1,i));
    temp2=0.1*(A(2,i-1)-B(1,i-1));
    temp3=0.1*(A(2,i+1)-B(1,i+1));
    temp4=0.05*(A(2,i-2)-B(1,i-2));
    temp5=0.05*(A(2,i+2)-B(1,i+2));
    tempError(i)=abs(temp1+temp2+temp3+temp4+temp5);
        if tempError(i)>tao
            Error(i)=1;
        else
            Error(i)=0;
        end  
end
degree=sum(Error);
end

