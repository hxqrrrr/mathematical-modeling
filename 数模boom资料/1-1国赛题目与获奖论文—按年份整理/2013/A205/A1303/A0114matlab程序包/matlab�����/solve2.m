%����solve1������ezplot�����ߣ���������ezplot('solve2(19.14,25,46,t)',[0,20]);
%�õ������Ŷӳ��Ƚ���������·�ڵ�ʱ�䣬��ͼ����ߵ�
function m=solve2(u,w,n,t)
%��������ĺ��壺
     %u��ʾ��λʱ��ͨ���ĳ�����mu����������ɵĶ������.����19.14
     %w��ʾ��λʱ�䵽�ﳵ����lambda.����25.
     %n��ʾ�Ŷӳ��ȡ�����46.
     %t��ʾ�Ŷ�ʱ��
 %��������ĺ��壺����
a0=1;
%����ֵ
%ͨ����ѭ�����a(n-1,,n)
for f=1:n
    e(f)=sqrt(4*w*u)*cos(f*pi/(n+1));
    a(1,f)=e(f)/(e(f)-u);
    for i=2:n-1
        a(i,f)=(w-e(f)-(w/a(i-1,f)))/(w+u-e(f)-(w/a(i-1,f)));
    end
    for i=2:n
         s=a0;
         p=1;
        for k=1:i-1
            s =a(k,f)*s ;
        end
        if i<n
        for o=1:i
            p =(a(o,f)-1)*p;
        end
        else
             for o=1:i-1
                 p =(a(o,f)-1)*p;
             end
        end      
        rr(i,f)=s/p;
    end
    rr(1,f)=a0/(a(1,f)-1);
end
rr(n,:)=-rr(n,:);

%���x����֪����Q������ֵ
%�������Q
Q=zeros(n+1,n+1);
Q(1,1)=-w;
Q(1,2)=u;
Q(n+1,n)=w;
Q(n+1,n+1)=-u;
for i=1:n-1
    Q(i+1,i)=w;
    Q(i+1,i+1)=-w-u;
    Q(i+1,i+2)=u;
end

%��matlab�����亯��eig������ֵx,��������r
x=eig(Q);
[r,aa]=eig(Q);
%�������B
B(1)=1;
for i=1:n
    B(i+1)=0;
end
B=B';
%�����Է���r*c=B
c=r\B;
c=c';

%su=c(1)*(w/u)^n;
%ѭ����ƽ���ӳ��ﵽnֵ�ĸ���
ss=0; 
    for k=1:n+1
        ss=c(k)*2.718^(x(k)*t)*r(46,k)+ss;
    end
m=ss;
%�õ������ƽ���ӳ��ﵽnֵ�ĸ���Ϊm
end
       
