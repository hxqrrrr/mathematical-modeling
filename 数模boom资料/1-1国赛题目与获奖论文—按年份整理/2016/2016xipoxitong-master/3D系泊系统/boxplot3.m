function boxplot3(x0,y0,z0,Lx,Ly,Lz) 
%�˺������ڻ���������
%
%%%%����˵��%%%%
%(x0,y0,z0)�ǵ�һ�������λ��; 
%(Lx,Ly,Lz)�ǳ�����ĳ����.
%

%%%%����%%%%
x=[x0 x0 x0 x0 x0+Lx x0+Lx x0+Lx x0+Lx];
y=[y0 y0 y0+Ly y0+Ly y0 y0 y0+Ly y0+Ly];
z=[z0 z0+Lz z0+Lz z0 z0 z0+Lz z0+Lz z0];
index=zeros(6,5);
index(1,:)=[1 2 3 4 1];
index(2,:)=[5 6 7 8 5];
index(3,:)=[1 2 6 5 1];
index(4,:)=[4 3 7 8 4];
index(5,:)=[2 6 7 3 2];
index(6,:)=[1 5 8 4 1];
for k=1:6
    fill3(x(index(k,:)),y(index(k,:)),z(index(k,:)), 'k')
    hold on
end
end