function B=jz01zh(A)
%���Ҷ�ֵ����Aת��Ϊ0-1����B
[rows cols]=size(A);
for i=1:rows
    for j=1:cols
        if A(i,j)<255
            B(i,j)=0;
        end
        if A(i,j)==255
            B(i,j)=1;
        end
    end
end

