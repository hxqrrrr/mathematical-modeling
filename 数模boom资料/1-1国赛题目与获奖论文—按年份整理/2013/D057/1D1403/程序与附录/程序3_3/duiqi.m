function y=duiqi(x)
m=[1:181]';%m�ĵ�һ��Ϊվ��,�Ժ����Ϊ������¸�վͳ����
for i=1:size(x,1)
    for j=1:size(x,2)/2
        if x(i,j*2-1)~=0
        m(x(i,j*2-1),j+1)=x(i,j*2);
        end
    end
end
y=m;