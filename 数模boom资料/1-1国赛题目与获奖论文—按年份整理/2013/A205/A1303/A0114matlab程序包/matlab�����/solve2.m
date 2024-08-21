%调用solve1函数用ezplot画曲线，输入命令ezplot('solve2(19.14,25,46,t)',[0,20]);
%得到车辆排队长度将到达上游路口的时间，即图像最高点
function m=solve2(u,w,n,t)
%输入参数的含义：
     %u表示单位时间通过的车辆数mu，即服务完成的对象个数.输入19.14
     %w表示单位时间到达车辆数lambda.输入25.
     %n表示排队长度。输入46.
     %t表示排队时间
 %输出参数的含义：概率
a0=1;
%赋初值
%通过大循环求解a(n-1,,n)
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

%求解x，已知矩阵Q的特征值
%构造矩阵Q
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

%用matlab工具箱函数eig求特征值x,特征向量r
x=eig(Q);
[r,aa]=eig(Q);
%构造矩阵B
B(1)=1;
for i=1:n
    B(i+1)=0;
end
B=B';
%解线性方程r*c=B
c=r\B;
c=c';

%su=c(1)*(w/u)^n;
%循环求平均队长达到n值的概率
ss=0; 
    for k=1:n+1
        ss=c(k)*2.718^(x(k)*t)*r(46,k)+ss;
    end
m=ss;
%得到结果，平均队长达到n值的概率为m
end
       
