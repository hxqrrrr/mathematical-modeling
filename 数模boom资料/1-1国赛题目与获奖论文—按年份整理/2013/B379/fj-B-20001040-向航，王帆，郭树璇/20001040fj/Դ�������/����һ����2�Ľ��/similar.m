function [ similarity ] = similar( A,B )
%similar is to calculate the similarity of two picture.
%   A,B is the edge of two picture.

n=length(A);
ratio=zeros;
for i=1:n
    if A(i,2)==0||B(i,1)==0
    ratio(i)=0;
    continue;
    end
    ratio(i)=min(A(i,2),B(i,1))/max(A(i,2),B(i,1));
end

similarity=sum(ratio);

end

