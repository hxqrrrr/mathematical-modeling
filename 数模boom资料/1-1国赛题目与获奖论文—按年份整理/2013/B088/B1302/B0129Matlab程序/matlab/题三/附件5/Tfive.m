clear all;clc;
load 'hang1.mat';
load 'hang2.mat';
load 'tu.mat';
hang=load('hang.txt');
for i = 1:418
    tu{i,1} = double(tu{i,1}); %ת��Ϊdouble��
end
bm=tu;

for r=1:11
if(r==6)
    continue;   %��6�д���ܶ࣬Ӧȫ�����˹���Ԥ
end
ju=hang(r,:);
n=length(ju);
W=inf*ones(n,n);
afa=1;beta=1;
for i=1:n
    for j=1:n 
        W(i,j)=0;
        w3=(hang1(i)-hang1(j))^2/180;
        w4=(hang2(i)-hang2(j))^2/180;
        for k=1:180
        w1=(bm{ju(i),1}(k,72)-afa*bm{ju(j),1}(k,1)-beta)^2/180;   %������
        
        if((mod(ju(i),2)==1)&(mod(ju(j),2)==1))
        w2=(bm{ju(j)+1,1}(k,72)-afa*bm{ju(i)+1+1,1}(k,1)-beta)^2/180;
        
        elseif((mod(ju(i),2)==1)&(mod(ju(j),2)==0))
        w2=(bm{ju(j)-1,1}(k,72)-afa*bm{ju(i)+1,1}(k,1)-beta)^2/180; 
        
        elseif((mod(ju(i),2)==0)&(mod(ju(j),2)==1))
        w2=(bm{ju(j)+1,1}(k,72)-afa*bm{ju(i)-1,1}(k,1)-beta)^2/180; 
        
        elseif((mod(ju(i),2)==0)&(mod(ju(j),2)==0))
        w2=(bm{ju(j)-1,1}(k,72)-afa*bm{ju(i)-1,1}(k,1)-beta)^2/180; 
        end
        W(i,j)=W(i,j)+w1+w2+w3+w4;
        end
    end
end


n=1;
for ci=1:18
  k=find(W(n,:)==min(W(n,:)));
  if(k==n)
      W(n,k)=inf;
      k=find(W(n,:)==min(W(n,:)));
  end
  k=k(1);
  be(r,ci)=ju(k);
   W(n,k)=inf;
    n=k;
    for i=1:19
        W(i,n)=inf;
    end
end
end
%be��11��18������Ϊ���г������������