%%%%问题4敏感分析
function pro4
for i=(1:8)
QQ(i)=prob4(30*i);
end
bar(QQ)
end


function  Y=prob4(T)
c=5*T/12;
na=0.5;
Q1=zeros(1,600); %上游累计到达车辆数
Q2=zeros(1,600); %下游离开车辆
for t=1:600
    n=ceil(t/T);
    t0=rem(t,T)/T;
    k=c*(1+na);
    temp=c/k;
    if t0<temp;
        Q1(t)=k/T;
    else
        Q1(t)=0;
    end
    Q2(t)=16.6/60;
end
Q3=cumsum(Q1);
Q4=cumsum(Q2);

i=(1:600)/60;
S=2.63*(Q3-Q4);
%plot(i,S),hold on,
%plot([0,10],[140,140],'r:'),xlabel('时间/min'),ylabel('排队长度/m');
Z=find((Q3-Q4)>140/2.63);
Y=Z(1);
end