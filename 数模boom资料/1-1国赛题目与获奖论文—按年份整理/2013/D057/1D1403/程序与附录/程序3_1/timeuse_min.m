clc;clear;
data=xlsread('day_20.xls');%��һ���ǽ賵վ�ţ��ڶ����ǻ���վ�ţ����������ó�ʱ��
%PΪ������������Сʱ��
min_time=ones(181)*inf;
for i=1:size(data,1)-1
    if min_time(data(i,1),data(i,2))==inf
        min_time(data(i,1),data(i,2))=data(i,3);
    end
end
min_time;
for i=1:181
    for j=i:181
        min_time(i,j)=min(min_time(i,j),min_time(j,i));
        min_time(j,i)=min(min_time(i,j),min_time(j,i));
    end
end

P=floyd(min_time)
xlswrite('day_20_P.xls',P)