function output=grayrela
x0=xlsread('�ο�������Ƚ�����.xls','Sheet1','B2:F182')'
%�����ɫ����ϵ��
%�ο�������Ƚ����ӹ�ͬ�洢��һ������x0��,�ο�����λ�ڵ�һ��

%б������
for i=2:length(x0(:,1))
    x1(i,:)=x0(i,:)-x0(i-1,:);
end

%��׼��
m=length(x1(1,:));
for i=1:m
x2(:,i)=x1(:,i)/std(x1(:,i));
end

%����
[y,pos]=sort(x2(:,1));
x2_sorted=x2(pos,:);

% �ж���������
n=length(x1(:,1));
k=[1:n]';
for j=1:m
sig_j(j)=qiuhe(k.*x2_sorted(:,j))-qiuhe(x2_sorted(:,j))*qiuhe(k)/n;
end

%caculation of distantion
for j=2:m
    dist_0i(:,j)=abs(sign(sig_j(:,j)./sig_j(:,1)).*x2_sorted(:,j)-x2_sorted(:,1));
end

%�������ϵ��
for i=1:n
    for j=1:m
        coef_rela(i,j)=(min(dist_0i)+0.5*max(dist_0i))/(dist_0i(i,j)+0.5*max(dist_0i));
    end
end

for j=1:m
    output(j)=qiuhe(coef_rela(:,j))/n;
end
xlswrite('day_20_Coef',output);
% function output=qiuhe(input)
% output=0;
% for i=1:length(input)
%     output=output+input(i);
% end

